"""Utility riutilizzabili per il notebook 03_transfer_learning.

Il modulo non definisce il disegno sperimentale: riceve esplicitamente dal
notebook configurazione, dataframe, file sorgente e accumulatori dei risultati.
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
    """Legge un CSV opzionale senza fallire su file assenti o vuoti."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def add_fold_key(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge una chiave temporanea che rende equivalenti i fold mancanti."""
    result = df.copy()
    if "fold" not in result.columns:
        result["fold"] = None

    def normalize_fold(value: Any) -> str:
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
    """Concatena due tabelle e conserva l'ultima riga per chiave logica."""
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
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(
            f"{file_path}: colonne mancanti {absent}. Controllare il CSV di Persona 1."
        )


def validate_common(
    df: pd.DataFrame,
    file_path: Path | str,
    dataset_config: str,
    require_split: bool = False,
    require_fold: bool = False,
) -> None:
    required = set(BASE_COLUMNS)
    if require_split:
        required.add("split")
    if require_fold:
        required.add("fold")
    require_columns(df, required, file_path)

    configs = set(df["dataset_config"].dropna().astype(str).unique())
    if configs != {dataset_config}:
        raise ValueError(
            f"{file_path}: dataset_config={configs}, atteso solo {dataset_config!r}."
        )
    labels = set(pd.to_numeric(df["binary_label"], errors="raise").unique())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError(
            f"{file_path}: binary_label deve contenere solo 0 e 1; trovati {labels}."
        )
    if require_split:
        splits = set(df["split"].dropna().astype(str).str.lower().unique())
        if splits != {"train", "val", "test"}:
            raise ValueError(
                f"{file_path}: split trovati {splits}; attesi train/val/test."
            )
    df["binary_label"] = df["binary_label"].astype(int)


def assert_patient_disjoint(df: pd.DataFrame, context: str) -> None:
    patient_sets = {
        split: set(df.loc[df["split"] == split, "patient_id"].astype(str))
        for split in ["train", "val", "test"]
    }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
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
    raw_path = row.get("path")
    raw_relative = row.get("relative_path")
    cache_key = (str(raw_path), str(raw_relative))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    candidates: list[Path] = []
    if pd.notna(raw_relative):
        rel = Path(str(raw_relative))
        if rel.is_absolute():
            candidates.append(rel)
        else:
            candidates.extend(
                [
                    dataset_root / rel,
                    project_root / rel,
                    project_root / "data" / "original" / rel,
                    project_root / "data" / "original" / "BreaKHis_v1" / rel,
                ]
            )
    # Il path assoluto del CSV può riferirsi all'ambiente di preprocessing:
    # viene usato solo come fallback dopo DATASET_ROOT / relative_path.
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
    if not config["FAST_DEV_RUN"]:
        return df.copy().reset_index(drop=True)
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
    paths = df["resolved_path"].astype(str).to_numpy()
    labels = df["binary_label"].astype(np.float32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            len(df), seed=config["SEED"], reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda path, label: decode_resize(
            path, label, preprocessing, config["IMAGE_SIZE"]
        ),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(config["BATCH_SIZE"]).prefetch(AUTOTUNE)


def get_preprocess_fn(backbone_name: str) -> Any:
    if backbone_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if backbone_name == "efficientnetb0":
        # In Keras corrente è pass-through: la normalizzazione è nel backbone.
        return tf.keras.applications.efficientnet.preprocess_input
    raise ValueError(f"Backbone non supportato: {backbone_name}")


def get_backbone_constructor(backbone_name: str) -> Any:
    constructors = {
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
    }
    if backbone_name not in constructors:
        raise ValueError(f"Backbone non supportato: {backbone_name}")
    return constructors[backbone_name]


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
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
    backbone.trainable = True
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
    if training_mode not in {"frozen", "finetuned"}:
        raise ValueError("training_mode deve essere 'frozen' o 'finetuned'.")
    constructor = get_backbone_constructor(backbone_name)
    preprocessing = get_preprocess_fn(backbone_name)
    image_size = config["IMAGE_SIZE"]
    backbone = constructor(
        weights="imagenet",
        include_top=False,
        input_shape=(*image_size, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config["DROPOUT_RATE"])(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignant_probability"
    )(x)
    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_binary")
    compile_model(model, config["FROZEN_LR"])
    if training_mode == "finetuned":
        enable_partial_finetuning(model, backbone, config)
    return model, backbone, preprocessing


class EpochTimer(tf.keras.callbacks.Callback):
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
    timer = EpochTimer()
    callbacks = [
        timer,
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
    total = int(model.count_params())
    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
    return total, trainable


def json_ready(value: Any) -> Any:
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
    cached_batches = [
        batch_images
        for batch_images, _ in test_ds.take(config["MODEL_ONLY_TIMING_MAX_BATCHES"])
    ]
    if not cached_batches:
        raise ValueError("Test dataset vuoto: impossibile misurare l'inferenza.")
    n_images = int(
        sum(int(tf.shape(batch)[0].numpy()) for batch in cached_batches)
    )

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
    for batch_images, _ in test_ds.take(1):
        _ = model(batch_images, training=False).numpy()

    started = time.perf_counter()
    y_prob = model.predict(test_ds, verbose=0).reshape(-1)
    inference_sec = time.perf_counter() - started
    end_to_end_ms_per_image = 1000 * inference_sec / max(1, len(test_df))
    model_only_timing = measure_model_only_inference(model, test_ds, config)

    y_true = test_df["binary_label"].to_numpy(dtype=int)
    y_pred = (y_prob >= 0.5).astype(int)
    if len(y_true) != len(y_prob):
        raise RuntimeError(
            f"Predizioni ({len(y_prob)}) e righe test ({len(y_true)}) non coincidono."
        )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auroc = (
        roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    )
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
    for path in output_paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training(
    log: pd.DataFrame, title: str, output_paths: list[Path]
) -> None:
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
    for path in output_paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
    """Restituisce la run completa più recente con la stessa identità sperimentale."""
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
    """Ricarica artefatti esistenti negli accumulatori senza rieseguire il training."""
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
    run_dir = experiment_dir(
        split_type, model_name, training_mode, config, fold=fold
    )
    model_path = run_dir / "model.keras"
    if config["SAVE_MODELS"]:
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
    if config.get("SKIP_EXISTING_RUNS", False):
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
    parts = {
        split: df[df["split"] == split].reset_index(drop=True)
        for split in ["train", "val", "test"]
    }
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{split_type}, fold={fold}: split {name} vuoto.")
        if part["binary_label"].nunique() < 2:
            warnings.warn(
                f"{split_type}, fold={fold}, {name}: una sola classe; "
                "FAST_DEV_RUN può invalidare AUROC."
            )

    model, backbone, preprocessing = build_transfer_model(
        backbone_name, training_mode="frozen", config=config
    )
    datasets = {
        split: make_dataset(
            part, preprocessing, config, training=(split == "train")
        )
        for split, part in parts.items()
    }
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

    frozen_log, frozen_time = fit_stage(
        model,
        datasets["train"],
        datasets["val"],
        config["FROZEN_EPOCHS"],
        config,
    )
    frozen_log["phase"] = "frozen"
    frozen_log["global_epoch"] = frozen_log["epoch"]
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
    tf.keras.backend.clear_session()
    return pd.DataFrame(metrics_store[-2:])


def find_baseline_table(
    filename: str, candidate_directories: list[Path]
) -> Path | None:
    return next(
        (
            directory / filename
            for directory in candidate_directories
            if (directory / filename).is_file()
        ),
        None,
    )
