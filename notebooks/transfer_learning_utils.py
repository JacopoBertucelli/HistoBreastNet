"""Utility riutilizzabili per il notebook 03_transfer_learning.

Il modulo non definisce il disegno sperimentale: riceve esplicitamente dal
notebook configurazione, dataframe, file sorgente e accumulatori dei risultati.
Gestisce validazione dei CSV di input, risoluzione dei path immagine,
costruzione dei dataset TensorFlow, creazione dei modelli di transfer learning,
training frozen e fine-tuning, valutazione, salvataggio di modelli, metriche,
predizioni e figure, oltre al riuso di run già completate.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, MutableSequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Colonne minime attese nei CSV prodotti dal preprocessing: garantiscono che il
# notebook 03 lavori su metadata coerenti. `relative_path` ricostruisce il path
# immagine, `binary_label` è il target binario, mentre `subtype_name`,
# `patient_id` e `magnification` servono per analisi successive. `dataset_config`
# evita di mescolare subset generati con configurazioni diverse.
BASE_COLUMNS = {
    "relative_path",
    "binary_label",
    "subtype_name",
    "patient_id",
    "magnification",
    "dataset_config",
}
AUTOTUNE = tf.data.AUTOTUNE


def safe_read_csv(path: Path | str) -> pd.DataFrame:
    """Legge un CSV opzionale in modo robusto.

    Ritorna un DataFrame vuoto se il file non esiste, ha dimensione zero o non
    contiene righe leggibili. Questo permette al notebook 03 di caricare
    artefatti già esistenti senza interrompersi quando una run non ha ancora
    prodotto tutti i file.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def add_fold_key(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge `_fold_key` normalizzando fold mancanti o non k-fold.

    La chiave temporanea consente di deduplicare nello stesso modo risultati da
    split singoli e da k-fold: fold assente, NaN, None o stringa vuota vengono
    trattati come lo stesso valore logico.
    """
    result = df.copy()
    if "fold" not in result.columns:
        result["fold"] = None

    def normalize_fold(value: Any) -> str:
        # Split singoli e fold mancanti devono collassare nella stessa chiave,
        # altrimenti la deduplicazione distinguerebbe artificialmente None, NaN
        # e stringhe vuote.
        if pd.isna(value):
            return "NO_FOLD"
        text = str(value).strip()
        if text.lower() in {"", "none", "nan", "<na>"}:
            return "NO_FOLD"
        try:
            numeric = float(text)
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass
        return text

    result["_fold_key"] = result["fold"].map(normalize_fold).astype("string")
    return result


def concat_and_deduplicate(
    existing_df: pd.DataFrame | None,
    current_df: pd.DataFrame | None,
    dedupe_cols: list[str],
) -> pd.DataFrame:
    """Concatena due tabelle e conserva l'ultima riga per chiave logica.

    È usata quando il notebook viene rieseguito parzialmente: nuovi risultati
    sostituiscono quelli precedenti per la stessa identità sperimentale, senza
    duplicare metriche o predizioni aggregate.
    """
    inputs = [df for df in [existing_df, current_df] if df is not None]
    frames = [df for df in inputs if not df.empty]
    if not frames:
        columns: list[str] = []
        for frame in inputs:
            columns.extend(column for column in frame.columns if column not in columns)
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "_fold_key" in dedupe_cols:
        combined = add_fold_key(combined)
    available_cols = [column for column in dedupe_cols if column in combined.columns]
    if available_cols:
        combined = combined.drop_duplicates(subset=available_cols, keep="last")
    return (
        combined.drop(columns=["_fold_key"], errors="ignore")
        .reset_index(drop=True)
    )


def require_columns(df: pd.DataFrame, required: set[str], file_path: Path | str) -> None:
    """Verifica che un CSV contenga tutte le colonne necessarie al protocollo."""
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(
            f"{file_path}: colonne mancanti {absent}. Controllare il CSV di preprocessing."
        )


def validate_common(
    df: pd.DataFrame,
    file_path: Path | str,
    dataset_config: str,
    require_split: bool = False,
    require_fold: bool = False,
) -> None:
    """Valida struttura e coerenza dei CSV di preprocessing.

    Impedisce di usare input incompleti o appartenenti a una configurazione di
    subset diversa da quella richiesta. Controlla inoltre che `binary_label` sia
    binario e, quando richiesto, che gli split siano esattamente train/val/test.
    """
    required = set(BASE_COLUMNS)
    if require_split:
        required.add("split")
    if require_fold:
        required.add("fold")
    require_columns(df, required, file_path)

    # `dataset_config` è un vincolo di riproducibilità: evita di unire risultati
    # o metadata prodotti da subset diversi.
    configs = set(df["dataset_config"].dropna().astype(str).unique())
    if configs != {dataset_config}:
        raise ValueError(
            f"{file_path}: dataset_config={configs}, atteso solo {dataset_config!r}."
        )
    # Il target del progetto è binario benign/malignant, codificato come 0/1.
    labels = set(pd.to_numeric(df["binary_label"], errors="raise").unique())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError(
            f"{file_path}: binary_label deve contenere solo 0 e 1; trovati {labels}."
        )
    if require_split:
        # Il notebook 03 assume sempre i tre ruoli canonici per training,
        # validation e test.
        splits = set(df["split"].dropna().astype(str).str.lower().unique())
        if splits != {"train", "val", "test"}:
            raise ValueError(
                f"{file_path}: split trovati {splits}; attesi train/val/test."
            )
    df["binary_label"] = df["binary_label"].astype(int)


def assert_patient_disjoint(df: pd.DataFrame, context: str) -> None:
    """Verifica l'assenza di leakage patient-wise nello split.

    Nessun `patient_id` deve comparire contemporaneamente in train, validation e
    test. Questo controllo è metodologicamente centrale perché la valutazione
    deve stimare la generalizzazione su pazienti nuovi, non su immagini correlate
    dello stesso paziente.
    """
    patient_sets = {
        split: set(df.loc[df["split"] == split, "patient_id"].astype(str))
        for split in ["train", "val", "test"]
    }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        # Qualunque overlap indica leakage: immagini dello stesso paziente
        # influenzerebbero contemporaneamente training e valutazione.
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise ValueError(
                f"{context}: overlap patient_id tra {left} e {right}: "
                f"{sorted(overlap)[:5]}."
            )


def resolve_image_path(
    row: pd.Series,
    dataset_root: Path,
    project_root: Path,
    cache: dict[tuple[str, str], str] | None = None,
) -> str:
    """Ricostruisce il path immagine in modo portabile.

    La priorità va a `relative_path`, più robusto tra computer locali, Colab e
    repository copiati. Il path assoluto salvato nel CSV viene usato solo come
    fallback, perché può riferirsi all'ambiente in cui è stato fatto il
    preprocessing. Una cache opzionale evita di ripetere gli stessi controlli su
    filesystem per righe con path identici.
    """
    raw_path = row.get("path")
    raw_relative = row.get("relative_path")
    # La cache usa sia path assoluto sia relative_path perché i CSV possono
    # contenere uno o entrambi; così si evitano risoluzioni ripetute.
    cache_key = (str(raw_path), str(raw_relative))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    candidates: list[Path] = []
    if pd.notna(raw_relative):
        rel = Path(str(raw_relative))
        if rel.is_absolute():
            candidates.append(rel)
        else:
            # Ordine di portabilità: dataset_root esplicito, path relativo al
            # progetto, copia sotto data/original e layout BreakHis standard.
            candidates.extend(
                [
                    dataset_root / rel,
                    project_root / rel,
                    project_root / "data" / "original" / rel,
                    project_root / "data" / "original" / "BreaKHis_v1" / rel,
                ]
            )
    # Il path assoluto del CSV può riferirsi all'ambiente di preprocessing:
    # viene usato solo come fallback dopo i candidati basati su relative_path.
    if pd.notna(raw_path):
        candidates.append(Path(str(raw_path)).expanduser())

    for candidate in candidates:
        if candidate.is_file():
            resolved = str(candidate.resolve())
            if cache is not None:
                cache[cache_key] = resolved
            return resolved
    raise FileNotFoundError(
        "Immagine non trovata. Controlla DATASET_ROOT, path e relative_path. "
        f"Primi candidati provati: {[str(candidate) for candidate in candidates[:5]]}"
    )


def print_split_statistics(df: pd.DataFrame, title: str) -> None:
    """Stampa statistiche rapide per controllare la qualità degli split.

    Riporta numero immagini, numero pazienti, distribuzione per classe,
    sottotipo e magnification. Questi controlli aiutano a intercettare squilibri
    o split anomali prima del training.
    """
    print(f"\n===== {title} =====")
    for split_name, part in df.groupby("split", sort=False):
        print(
            f"\n--- {split_name}: {len(part)} immagini, "
            f"{part.patient_id.nunique()} pazienti ---"
        )
        print(
            "Immagini per classe:\n",
            part["binary_label"].value_counts().sort_index().to_string(),
        )
        patients_by_class = part.groupby("binary_label")["patient_id"].nunique()
        print("Pazienti per classe:\n", patients_by_class.to_string())
        print(
            "Immagini per sottotipo:\n",
            part["subtype_name"].value_counts().to_string(),
        )
        print(
            "Immagini per magnification:\n",
            part["magnification"].value_counts().sort_index().to_string(),
        )


def dev_subset(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Riduce il DataFrame solo per test rapidi della pipeline.

    `FAST_DEV_RUN` non rappresenta la configurazione sperimentale finale: tiene
    pochi esempi per split e classe per verificare caricamento, training,
    valutazione e salvataggi senza attendere una run completa.
    """
    if not config["FAST_DEV_RUN"]:
        return df.copy().reset_index(drop=True)
    # Il campionamento resta stratificato per split/classe, sufficiente per
    # controllare che il flusso end-to-end del notebook 03 sia eseguibile.
    return (
        df.groupby(["split", "binary_label"], group_keys=False)
        .head(max(1, config["FAST_SAMPLES_PER_SPLIT"] // 2))
        .reset_index(drop=True)
    )


def decode_resize(
    path: tf.Tensor,
    label: tf.Tensor,
    preprocessing: Any,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Legge, ridimensiona e preprocessa un'immagine per il backbone scelto.

    L'immagine viene decodificata sempre in RGB a 3 canali, ridimensionata alla
    risoluzione richiesta dal backbone e trasformata con la funzione di
    preprocessing coerente con MobileNetV2 o EfficientNetB0. La label è float
    perché la loss usata nel task binario è `binary_crossentropy`.
    """
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = preprocessing(tf.cast(image, tf.float32))
    return image, tf.cast(label, tf.float32)


def make_dataset(
    df: pd.DataFrame,
    preprocessing: Any,
    config: dict[str, Any],
    training: bool = False,
) -> tf.data.Dataset:
    """Costruisce la pipeline `tf.data` da path risolti e label binarie."""
    paths = df["resolved_path"].astype(str).to_numpy()
    labels = df["binary_label"].astype(np.float32).to_numpy()
    # Punto di ingresso della pipeline TensorFlow: ogni elemento contiene path
    # immagine e target binario già validati dal preprocessing.
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        # Lo shuffle è applicato solo al training per ridurre correlazioni tra
        # batch, lasciando validation/test deterministici.
        dataset = dataset.shuffle(
            len(df), seed=config["SEED"], reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda path, label: decode_resize(
            path, label, preprocessing, config["IMAGE_SIZE"]
        ),
        num_parallel_calls=AUTOTUNE,
    )
    # Batch e prefetch tengono alimentata la GPU/CPU durante il training senza
    # cambiare il contenuto del dataset.
    return dataset.batch(config["BATCH_SIZE"]).prefetch(AUTOTUNE)


def get_preprocess_fn(backbone_name: str) -> Any:
    """Restituisce il preprocessing corretto per il backbone selezionato.

    Ogni architettura pre-addestrata richiede input coerenti con il proprio
    training ImageNet. MobileNetV2 usa il suo `preprocess_input`; EfficientNetB0
    nelle versioni Keras recenti integra la normalizzazione nel modello, quindi
    la funzione è di fatto pass-through.
    """
    if backbone_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if backbone_name == "efficientnetb0":
        # In Keras corrente è pass-through: la normalizzazione è nel backbone.
        return tf.keras.applications.efficientnet.preprocess_input
    raise ValueError(f"Backbone non supportato: {backbone_name}")


def get_backbone_constructor(backbone_name: str) -> Any:
    """Centralizza i constructor Keras dei backbone supportati dal notebook 03."""
    constructors = {
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
    }
    if backbone_name not in constructors:
        raise ValueError(f"Backbone non supportato: {backbone_name}")
    return constructors[backbone_name]


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compila il modello per classificazione binaria benign/malignant.

    La testa sigmoid produce p(malignant), la loss è `binary_crossentropy` e le
    metriche monitorate durante train/validation sono accuracy e AUROC.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def enable_partial_finetuning(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    config: dict[str, Any],
) -> None:
    """Abilita il fine-tuning parziale del backbone.

    Vengono sbloccati solo gli ultimi layer, mentre quelli iniziali restano
    congelati per preservare feature generiche ImageNet. Le BatchNormalization
    rimangono congelate per maggiore stabilità sui batch relativamente piccoli.
    Il modello viene ricompilato con learning rate più basso: questa è la fase
    fine-tuned del confronto frozen vs fine-tuned.
    """
    backbone.trainable = True
    # Determina il punto da cui sbloccare i layer finali del backbone.
    cutoff = max(0, len(backbone.layers) - config["UNFREEZE_LAST_N"])
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    compile_model(model, config["FINETUNE_LR"])


def build_transfer_model(
    backbone_name: str,
    training_mode: str,
    config: dict[str, Any],
) -> tuple[tf.keras.Model, tf.keras.Model, Any]:
    """Costruisce un modello di transfer learning per il task binario.

    Crea MobileNetV2 o EfficientNetB0 con pesi ImageNet, rimuove la testa
    originale (`include_top=False`) e aggiunge una testa binaria composta da
    GlobalAveragePooling, Dropout e Dense sigmoid. Il backbone parte sempre
    congelato; se `training_mode == "finetuned"` viene subito abilitato il
    fine-tuning parziale. Restituisce modello, backbone e funzione di
    preprocessing.
    """
    if training_mode not in {"frozen", "finetuned"}:
        raise ValueError("training_mode deve essere 'frozen' o 'finetuned'.")
    constructor = get_backbone_constructor(backbone_name)
    preprocessing = get_preprocess_fn(backbone_name)
    image_size = config["IMAGE_SIZE"]
    backbone = constructor(
        # Pesi ImageNet: punto di partenza del transfer learning.
        weights="imagenet",
        # Senza testa ImageNet, perché la classificazione finale è benign/malignant.
        include_top=False,
        input_shape=(*image_size, 3),
    )
    # La fase frozen addestra solo la nuova testa classificativa.
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = backbone(inputs, training=False)
    # Converte le feature spaziali del backbone in un vettore compatto.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dropout sulla testa per ridurre overfitting sul subset.
    x = tf.keras.layers.Dropout(config["DROPOUT_RATE"])(x)
    # Singola uscita sigmoid: probabilità stimata della classe malignant.
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignant_probability"
    )(x)
    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_binary")
    compile_model(model, config["FROZEN_LR"])
    if training_mode == "finetuned":
        enable_partial_finetuning(model, backbone, config)
    return model, backbone, preprocessing


class EpochTimer(tf.keras.callbacks.Callback):
    """Callback Keras che misura il tempo impiegato da ogni epoca."""

    def on_train_begin(self, logs: dict | None = None) -> None:
        self.epoch_times: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        self.epoch_times.append(time.perf_counter() - self._start)


def fit_stage(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    """Esegue una fase di training e restituisce log epoche e tempo totale.

    EarlyStopping monitora `val_loss`, evita training inutile quando la
    validation peggiora e ripristina i pesi migliori. I tempi salvati servono al
    confronto efficienza/prestazioni tra backbone e modalità frozen/fine-tuned.
    """
    timer = EpochTimer()
    callbacks = [
        timer,
        # Ripristinare i pesi migliori rende la valutazione finale coerente con
        # la migliore epoca di validation della fase.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["PATIENCE"],
            restore_best_weights=True,
        ),
    ]
    started = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.perf_counter() - started
    log = pd.DataFrame(history.history)
    log.insert(0, "epoch", np.arange(1, len(log) + 1))
    log["epoch_time_sec"] = timer.epoch_times
    return log, elapsed


def parameter_counts(model: tf.keras.Model) -> tuple[int, int]:
    """Restituisce parametri totali e trainable per stimare la complessità."""
    total = int(model.count_params())
    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
    return total, trainable


def json_ready(value: Any) -> Any:
    """Converte Path e tipi numpy in valori serializzabili in `config.json`."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def measure_model_only_inference(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """Misura il costo di inferenza del solo modello.

    Esclude lettura, decoding e resize delle immagini usando batch già
    materializzati in memoria. Dopo un warm-up, ripete la misurazione più volte
    e salva la mediana per ottenere una stima più stabile del costo computazionale.
    """
    # I batch vengono materializzati una volta per escludere la pipeline di input
    # dalla misura model-only.
    cached_batches = [
        batch_images
        for batch_images, _ in test_ds.take(config["MODEL_ONLY_TIMING_MAX_BATCHES"])
    ]
    if not cached_batches:
        raise ValueError("Test dataset vuoto: impossibile misurare l'inferenza.")
    n_images = int(
        sum(int(tf.shape(batch)[0].numpy()) for batch in cached_batches)
    )

    # Warm-up: forza inizializzazione/tracing prima della misura temporale.
    for batch_images in cached_batches:
        _ = model(batch_images, training=False).numpy()

    per_image_times_ms = []
    for _ in range(config["MODEL_ONLY_TIMING_REPEATS"]):
        started = time.perf_counter()
        for batch_images in cached_batches:
            _ = model(batch_images, training=False).numpy()
        elapsed = time.perf_counter() - started
        per_image_times_ms.append(1000 * elapsed / n_images)

    return {
        "model_only_inference_time_ms_per_image": float(
            np.median(per_image_times_ms)
        ),
        "model_only_inference_time_ms_std": (
            float(np.std(per_image_times_ms, ddof=1))
            if len(per_image_times_ms) > 1
            else 0.0
        ),
        "model_only_timing_n_images": n_images,
        "model_only_timing_n_batches": len(cached_batches),
        "model_only_timing_repeats": config["MODEL_ONLY_TIMING_REPEATS"],
    }


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Valuta il modello sul test set e prepara metriche e predizioni.

    Calcola p(malignant), applica la soglia decisionale 0.5, produce accuracy,
    precision, recall, F1, AUROC, recall_malignant e confusion matrix. Misura
    sia il tempo end-to-end sia il tempo model-only, costruisce il DataFrame
    delle predizioni con metadata utili ai notebook 04 e 05 e restituisce
    metriche, predizioni e matrice di confusione.
    """
    # Warm-up prima della predict: riduce l'effetto di inizializzazione sulla
    # misura del tempo di inferenza.
    for batch_images, _ in test_ds.take(1):
        _ = model(batch_images, training=False).numpy()

    started = time.perf_counter()
    # Output sigmoid del modello: probabilità stimata della classe malignant.
    y_prob = model.predict(test_ds, verbose=0).reshape(-1)
    inference_sec = time.perf_counter() - started
    end_to_end_ms_per_image = 1000 * inference_sec / max(1, len(test_df))
    model_only_timing = measure_model_only_inference(model, test_ds, config)

    y_true = test_df["binary_label"].to_numpy(dtype=int)
    # Soglia decisionale fissa del protocollo: p(malignant) >= 0.5.
    y_pred = (y_prob >= 0.5).astype(int)
    if len(y_true) != len(y_prob):
        raise RuntimeError(
            f"Predizioni ({len(y_prob)}) e righe test ({len(y_true)}) non coincidono."
        )
    # Con labels=[0, 1], la matrice è [[TN, FP], [FN, TP]].
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auroc = (
        roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    )
    # Le metriche salvate descrivono prestazioni, errori e costo di inferenza
    # della run sperimentale.
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": auroc,
        "recall_malignant": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_test_images": len(test_df),
        "n_test_patients": test_df["patient_id"].nunique(),
        "inference_time_sec": inference_sec,
        "inference_time_ms_per_image": end_to_end_ms_per_image,
        "end_to_end_inference_time_ms_per_image": end_to_end_ms_per_image,
        "threshold": 0.5,
    }
    metrics.update(model_only_timing)
    # I metadata restano nelle predizioni perché i notebook 04 e 05 riusano
    # questi file per analisi degli errori, Grad-CAM e robustness.
    prediction_cols = [
        column
        for column in [
            "relative_path",
            "filename",
            "patient_id",
            "binary_label",
            "label",
            "subtype",
            "subtype_name",
            "magnification",
        ]
        if column in test_df.columns
    ]
    predictions = test_df[prediction_cols].reset_index(drop=True).copy()
    predictions["y_true"] = y_true
    predictions["y_prob"] = y_prob
    predictions["y_pred"] = y_pred
    return metrics, predictions, cm


def plot_confusion(
    cm: np.ndarray, title: str, output_paths: list[Path]
) -> None:
    """Salva la confusion matrix della singola run nei path indicati."""
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    for (row, column), value in np.ndenumerate(cm):
        ax.text(column, row, str(value), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benigno", "Maligno"],
        yticklabels=["Benigno", "Maligno"],
        xlabel="Predetto",
        ylabel="Reale",
        title=title,
    )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    # Ogni figura viene salvata sia nella directory della run sia nella cartella
    # aggregata sotto results/03_transfer_learning/figures.
    for path in output_paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training(
    log: pd.DataFrame, title: str, output_paths: list[Path]
) -> None:
    """Salva curve loss e AUROC per fasi frozen e fine-tuning."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for phase, part in log.groupby("phase", sort=False):
        axes[0].plot(part["global_epoch"], part["loss"], label=f"{phase} train")
        axes[0].plot(
            part["global_epoch"], part["val_loss"], "--", label=f"{phase} val"
        )
        axes[1].plot(part["global_epoch"], part["auc"], label=f"{phase} train")
        axes[1].plot(
            part["global_epoch"], part["val_auc"], "--", label=f"{phase} val"
        )
    axes[0].set(title="Loss", xlabel="Epoca")
    axes[1].set(title="AUROC", xlabel="Epoca")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    # Stesso grafico in due posizioni: artefatto della run e raccolta aggregata
    # per confronto tra esperimenti.
    for path in output_paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Una run è riusabile solo se contiene configurazione, log, metriche, predizioni
# e figure principali; così si evitano run parziali o corrotte.
REQUIRED_RUN_ARTIFACTS = {
    "config.json",
    "training_log.csv",
    "metrics.csv",
    "predictions.csv",
    "confusion_matrix.png",
    "training_curves.png",
}


def find_completed_run(
    split_type: str,
    model_name: str,
    training_mode: str,
    config: dict[str, Any],
    fold: int | None = None,
) -> Path | None:
    """Cerca la run completa più recente con la stessa identità sperimentale.

    L'identità include dataset_config, split_type, modello, training_mode e fold:
    se tutti gli artefatti richiesti sono presenti, il notebook può saltare il
    training già completato senza usare risultati incompleti.
    """
    fold_suffix = "" if fold is None else f"_fold{fold}"
    pattern = (
        f"*_{config['DATASET_CONFIG']}_{split_type}_"
        f"{model_name}_{training_mode}{fold_suffix}"
    )
    candidates = sorted(
        path
        for path in config["EXPERIMENTS_DIR"].glob(pattern)
        if path.is_dir()
        and all((path / artifact).is_file() for artifact in REQUIRED_RUN_ARTIFACTS)
    )
    return candidates[-1] if candidates else None


def load_completed_protocol(
    completed_runs: dict[str, Path],
    split_type: str,
    model_name: str,
    fold: int | None,
    config: dict[str, Any],
    metrics_store: MutableSequence[dict[str, Any]],
    predictions_store: MutableSequence[pd.DataFrame],
    experiment_dirs_store: MutableSequence[Path],
) -> pd.DataFrame:
    """Ricarica metriche e predizioni già salvate senza rieseguire training.

    È usata quando `SKIP_EXISTING_RUNS=True`: reinserisce negli accumulatori le
    colonne di contesto necessarie al notebook 03 (`experiment_id`,
    dataset_config, split_type, fold, model, training_mode) e restituisce le
    metriche delle fasi frozen e fine-tuned.
    """
    loaded_metrics: list[dict[str, Any]] = []
    known_experiments = {
        str(metrics.get("experiment_id"))
        for metrics in metrics_store
        if metrics.get("experiment_id") is not None
    }
    for training_mode in ["frozen", "finetuned"]:
        run_dir = completed_runs[training_mode]
        metrics_df = safe_read_csv(run_dir / "metrics.csv")
        if metrics_df.empty:
            raise ValueError(f"metrics.csv vuoto nella run completa: {run_dir}")
        metrics = metrics_df.iloc[0].to_dict()
        metrics["experiment_id"] = run_dir.name
        loaded_metrics.append(metrics)
        if run_dir.name not in known_experiments:
            metrics_store.append(metrics.copy())
            predictions = safe_read_csv(run_dir / "predictions.csv")
            # Le predizioni aggregate mantengono l'identità sperimentale per
            # confronti e riuso nei notebook successivi.
            predictions.insert(0, "experiment_id", run_dir.name)
            predictions.insert(1, "dataset_config", config["DATASET_CONFIG"])
            predictions.insert(2, "split_type", split_type)
            predictions.insert(3, "fold", fold)
            predictions.insert(4, "model", model_name)
            predictions.insert(5, "training_mode", training_mode)
            predictions_store.append(predictions)
            known_experiments.add(run_dir.name)
        if run_dir not in experiment_dirs_store:
            experiment_dirs_store.append(run_dir)
    return pd.DataFrame(loaded_metrics)


def experiment_dir(
    split_type: str,
    model_name: str,
    training_mode: str,
    config: dict[str, Any],
    fold: int | None = None,
) -> Path:
    """Crea una directory timestampata per una run sperimentale.

    Il nome include dataset_config, split_type, modello, training_mode e fold,
    evitando sovrascritture tra esperimenti con identità diversa.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fold_suffix = "" if fold is None else f"_fold{fold}"
    path = config["EXPERIMENTS_DIR"] / (
        f"{stamp}_{config['DATASET_CONFIG']}_{split_type}_"
        f"{model_name}_{training_mode}{fold_suffix}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run(
    model: tf.keras.Model,
    log: pd.DataFrame,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    cm: np.ndarray,
    run_config: dict[str, Any],
    split_type: str,
    model_name: str,
    training_mode: str,
    config: dict[str, Any],
    predictions_store: MutableSequence[pd.DataFrame],
    experiment_dirs_store: MutableSequence[Path],
    fold: int | None = None,
) -> dict[str, Any]:
    """Salva tutti gli artefatti prodotti da una run.

    In base alla configurazione salva `model.keras`, `metrics.csv`,
    `predictions.csv`, `training_log.csv`, `config.json`,
    `confusion_matrix.png` e `training_curves.png`. Aggiorna anche gli
    accumulatori usati dal notebook 03 per costruire tabelle aggregate.
    """
    run_dir = experiment_dir(
        split_type, model_name, training_mode, config, fold=fold
    )
    model_path = run_dir / "model.keras"
    if config["SAVE_MODELS"]:
        # I modelli .keras possono non essere versionati, ma sono utili per
        # Grad-CAM e robustness nei notebook successivi.
        model.save(model_path)
        metrics["model_size_mb"] = model_path.stat().st_size / (1024**2)
    else:
        metrics["model_size_mb"] = sum(
            np.prod(weight.shape) * tf.as_dtype(weight.dtype).size
            for weight in model.weights
        ) / (1024**2)
    metrics["model_size_is_estimate"] = not config["SAVE_MODELS"]
    metrics["experiment_id"] = run_dir.name

    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    aggregate_predictions = predictions.copy()
    # Le colonne inserite rendono le predizioni aggregabili tra dataset_config,
    # split, fold, backbone e modalità di training.
    aggregate_predictions.insert(0, "experiment_id", run_dir.name)
    aggregate_predictions.insert(1, "dataset_config", config["DATASET_CONFIG"])
    aggregate_predictions.insert(2, "split_type", split_type)
    aggregate_predictions.insert(3, "fold", fold)
    aggregate_predictions.insert(4, "model", model_name)
    aggregate_predictions.insert(5, "training_mode", training_mode)
    predictions_store.append(aggregate_predictions)
    log.to_csv(run_dir / "training_log.csv", index=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(run_config), handle, indent=2, ensure_ascii=False)

    key = f"{split_type}_{model_name}_{training_mode}"
    if fold is not None:
        key += f"_fold{fold}"
    # Figure salvate sia nella run sia nella cartella aggregata delle figure.
    plot_confusion(
        cm,
        key,
        [
            run_dir / "confusion_matrix.png",
            config["FIGURES_DIR"] / "confusion_matrices" / f"{key}.png",
        ],
    )
    plot_training(
        log,
        key,
        [
            run_dir / "training_curves.png",
            config["FIGURES_DIR"] / "training_curves" / f"{key}.png",
        ],
    )
    experiment_dirs_store.append(run_dir)
    return metrics


def run_protocol(
    source_df: pd.DataFrame,
    split_type: str,
    config: dict[str, Any],
    input_files: dict[str, Path],
    metrics_store: MutableSequence[dict[str, Any]],
    predictions_store: MutableSequence[pd.DataFrame],
    experiment_dirs_store: MutableSequence[Path],
    fold: int | None = None,
    backbone_name: str = "mobilenetv2",
) -> pd.DataFrame:
    """Esegue un protocollo sperimentale per uno split/fold e un backbone.

    La funzione può riusare run già completate, costruisce train/val/test,
    crea il modello frozen, addestra, valuta e salva la fase frozen, abilita il
    fine-tuning parziale, ripete training/valutazione/salvataggio per la fase
    fine-tuned, aggiorna metriche, predizioni e directory esperimenti e ritorna
    le metriche delle due fasi.
    """
    if config.get("SKIP_EXISTING_RUNS", False):
        # Riuso controllato: si salta il training solo se entrambe le fasi hanno
        # tutti gli artefatti richiesti.
        completed_runs = {
            training_mode: find_completed_run(
                split_type,
                backbone_name,
                training_mode,
                config,
                fold=fold,
            )
            for training_mode in ["frozen", "finetuned"]
        }
        if all(completed_runs.values()):
            print(
                "Run già completa, training saltato: "
                f"split={split_type}, fold={fold}, backbone={backbone_name}"
            )
            return load_completed_protocol(
                completed_runs,
                split_type,
                backbone_name,
                fold,
                config,
                metrics_store,
                predictions_store,
                experiment_dirs_store,
            )
        if any(completed_runs.values()):
            warnings.warn(
                "Trovata solo una fase completa per "
                f"split={split_type}, fold={fold}, backbone={backbone_name}. "
                "Il protocollo frozen + fine-tuning viene rieseguito integralmente."
            )

    df = dev_subset(source_df, config)
    # Suddivisione train/val/test già definita nei CSV di preprocessing; qui il
    # notebook 03 prepara le tre viste operative del protocollo.
    parts = {
        split: df[df["split"] == split].reset_index(drop=True)
        for split in ["train", "val", "test"]
    }
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{split_type}, fold={fold}: split {name} vuoto.")
        if part["binary_label"].nunique() < 2:
            # AUROC richiede entrambe le classi; il warning è particolarmente
            # utile nei test rapidi con FAST_DEV_RUN.
            warnings.warn(
                f"{split_type}, fold={fold}, {name}: una sola classe; "
                "FAST_DEV_RUN può invalidare AUROC."
            )

    # Prima fase: backbone congelato e sola testa classificativa addestrabile.
    model, backbone, preprocessing = build_transfer_model(
        backbone_name, training_mode="frozen", config=config
    )
    # Dataset TensorFlow condivisi dalle due fasi, con shuffle solo per train.
    datasets = {
        split: make_dataset(
            part, preprocessing, config, training=(split == "train")
        )
        for split, part in parts.items()
    }
    # Config comune salvata in ogni run per riprodurre input, split e parametri.
    common_config = {
        "dataset_config": config["DATASET_CONFIG"],
        "split_type": split_type,
        "fold": fold,
        "backbone": backbone_name,
        "image_size": config["IMAGE_SIZE"],
        "batch_size": config["BATCH_SIZE"],
        "seed": config["SEED"],
        "fast_dev_run": config["FAST_DEV_RUN"],
        "max_folds": config.get("MAX_FOLDS"),
        "folds_to_run": config.get("FOLDS_TO_RUN"),
        "skip_existing_runs": config.get("SKIP_EXISTING_RUNS", False),
        "threshold": 0.5,
        "model_only_timing_max_batches": config[
            "MODEL_ONLY_TIMING_MAX_BATCHES"
        ],
        "model_only_timing_repeats": config["MODEL_ONLY_TIMING_REPEATS"],
        "source_csv": str(
            input_files["kfold"] if fold is not None else input_files[split_type]
        ),
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
    }

    # Training frozen: impara la testa binaria sopra feature ImageNet congelate.
    frozen_log, frozen_time = fit_stage(
        model,
        datasets["train"],
        datasets["val"],
        config["FROZEN_EPOCHS"],
        config,
    )
    frozen_log["phase"] = "frozen"
    frozen_log["global_epoch"] = frozen_log["epoch"]
    # Valutazione sul test set del protocollo corrente.
    metrics, predictions, cm = evaluate_model(
        model, datasets["test"], parts["test"], config
    )
    total_params, trainable_params = parameter_counts(model)
    metrics.update(
        {
            "model": backbone_name,
            "training_mode": "frozen",
            "dataset_config": config["DATASET_CONFIG"],
            "split_type": split_type,
            "fold": fold,
            "n_params": total_params,
            "n_trainable_params": trainable_params,
            "training_time_sec": frozen_time,
            "avg_epoch_time_sec": frozen_log["epoch_time_sec"].mean(),
            "epochs_run": len(frozen_log),
            "n_train": len(parts["train"]),
            "n_val": len(parts["val"]),
        }
    )
    frozen_config = {
        **common_config,
        "training_mode": "frozen",
        "learning_rate": config["FROZEN_LR"],
        "max_epochs": config["FROZEN_EPOCHS"],
        "dropout_rate": config["DROPOUT_RATE"],
    }
    metrics_store.append(
        # Salvataggio artefatti frozen: metriche, predizioni, log, config e figure.
        save_run(
            model,
            frozen_log,
            metrics,
            predictions,
            cm,
            frozen_config,
            split_type,
            backbone_name,
            "frozen",
            config,
            predictions_store,
            experiment_dirs_store,
            fold,
        ).copy()
    )

    # Seconda fase: fine-tuning parziale degli ultimi layer del backbone.
    enable_partial_finetuning(model, backbone, config)
    ft_log, ft_time = fit_stage(
        model,
        datasets["train"],
        datasets["val"],
        config["FINETUNE_EPOCHS"],
        config,
    )
    ft_log["phase"] = "finetune"
    ft_log["global_epoch"] = len(frozen_log) + ft_log["epoch"]
    # Log cumulativo: conserva storia frozen + fine-tuning nella run finale.
    combined_log = pd.concat([frozen_log, ft_log], ignore_index=True)
    metrics, predictions, cm = evaluate_model(
        model, datasets["test"], parts["test"], config
    )
    total_params, trainable_params = parameter_counts(model)
    total_training_time = frozen_time + ft_time
    metrics.update(
        {
            "model": backbone_name,
            "training_mode": "finetuned",
            "dataset_config": config["DATASET_CONFIG"],
            "split_type": split_type,
            "fold": fold,
            "n_params": total_params,
            "n_trainable_params": trainable_params,
            "training_time_sec": total_training_time,
            "avg_epoch_time_sec": combined_log["epoch_time_sec"].mean(),
            "frozen_stage_time_sec": frozen_time,
            "finetune_stage_time_sec": ft_time,
            "epochs_run": len(combined_log),
            "n_train": len(parts["train"]),
            "n_val": len(parts["val"]),
        }
    )
    finetune_config = {
        **common_config,
        "training_mode": "finetuned",
        "frozen_learning_rate": config["FROZEN_LR"],
        "finetune_learning_rate": config["FINETUNE_LR"],
        "frozen_max_epochs": config["FROZEN_EPOCHS"],
        "finetune_max_epochs": config["FINETUNE_EPOCHS"],
        "unfreeze_last_n": config["UNFREEZE_LAST_N"],
        "batch_norm_frozen": True,
        "training_time_is_cumulative": True,
        "dropout_rate": config["DROPOUT_RATE"],
    }
    metrics_store.append(
        # Salvataggio artefatti fine-tuned, inclusa la storia cumulativa.
        save_run(
            model,
            combined_log,
            metrics,
            predictions,
            cm,
            finetune_config,
            split_type,
            backbone_name,
            "finetuned",
            config,
            predictions_store,
            experiment_dirs_store,
            fold,
        ).copy()
    )
    del model, backbone, datasets
    # Libera grafi e memoria Keras prima di passare alla run successiva.
    tf.keras.backend.clear_session()
    return pd.DataFrame(metrics_store[-2:])


def find_baseline_table(
    filename: str, candidate_directories: list[Path]
) -> Path | None:
    """Cerca una tabella della baseline CNN in più directory candidate.

    Il notebook 03 può essere eseguito in ambienti diversi o con risultati già
    copiati: la ricerca flessibile evita di codificare una sola posizione.
    """
    return next(
        (
            directory / filename
            for directory in candidate_directories
            if (directory / filename).is_file()
        ),
        None,
    )
