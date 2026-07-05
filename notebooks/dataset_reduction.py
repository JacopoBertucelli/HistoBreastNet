"""dataset_reduction.py — preprocessing BreakHis (v4).

Novità v4 (revisione Persona 2):
  - extract_metadata: aggiunge 'relative_path' (path relativo alla root del
    dataset, per ricostruire i path lato Persona 2) e filtra i file spazzatura
    (__MACOSX, ._*).
  - Organizzazione PER CONFIG: write_config_bundle() salva tutti i CSV di una
    configurazione in una cartella dedicata (es. data/processed/diversity_1p5GB/),
    con colonna 'dataset_config' in ogni file, statistiche complete e config.json.
  - full_statistics(): tabella unica con distribuzioni per classe/sottotipo/
    magnification + split image-wise/patient-wise + per-fold della k-fold.
  - file k-fold rinominato in 'patient_wise_folds.csv'.

Invarianti: pazienti sempre interi, tutte le magnification per paziente,
random_state fisso. Le funzioni v2/v3 restano compatibili.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

CLASS_MAP = {'B':'benign','M':'malignant'}
SUBTYPE_NAMES = {'A':'adenosis','F':'fibroadenoma','PT':'phyllodes_tumor','TA':'tubular_adenoma',
 'DC':'ductal_carcinoma','LC':'lobular_carcinoma','MC':'mucinous_carcinoma','PC':'papillary_carcinoma'}

# =====================================================================
# METADATA
# =====================================================================
def extract_metadata(source):
    """Metadati dalla struttura cartelle. patient_id = nome cartella-slide (unico).
    Aggiunge 'relative_path' (rispetto a `source`) e ignora __MACOSX / ._*."""
    source = Path(source)
    rows = []
    for path in source.rglob('*.png'):
        if '__MACOSX' in path.parts or path.name.startswith('._'):
            continue                                   # file spazzatura
        slide = path.parent.parent.name
        parts = slide.split('_')
        if len(parts) >= 4 and parts[1] in CLASS_MAP:
            mag = path.parent.name
            label = CLASS_MAP[parts[1]]
            rows.append(dict(
                path=str(path),
                relative_path=path.relative_to(source).as_posix(),
                filename=path.name,
                label=label, binary_label=1 if label == 'malignant' else 0,
                subtype=parts[2], subtype_name=SUBTYPE_NAMES.get(parts[2], parts[2]),
                patient_id=slide,
                magnification=mag if mag.endswith('X') else mag+'X',
                file_size_bytes=path.stat().st_size))
    df = pd.DataFrame(rows)
    assert not df.empty, f"Nessuna immagine valida sotto {source}"
    return df

# =====================================================================
# SUBSET BUILDERS
# =====================================================================
def build_subset(df, images_per_subtype=380, random_state=42):
    """Config 'cap380': cap uguale per sottotipo, a pazienti interi.
    Ritorna (selected, subset, manifest)."""
    rng = np.random.default_rng(random_state)
    pats = (df.groupby(['patient_id','label','subtype'], as_index=False)
              .agg(n_images=('filename','count'),
                   bytes=('file_size_bytes','sum')))
    pats['size_mb'] = pats['bytes']/1024**2
    sel = []
    for (label, st), grp in pats.groupby(['label','subtype']):
        cand = grp.sample(frac=1.0, random_state=int(rng.integers(0,2**31-1)))
        n = 0
        for _, r in cand.iterrows():
            sel.append(r.patient_id); n += r.n_images
            if n >= images_per_subtype:
                break
    sel = sorted(set(sel))
    subset = df[df.patient_id.isin(sel)].copy()
    manifest = pats[pats.patient_id.isin(sel)].sort_values(['label','subtype','patient_id'])
    assert subset['magnification'].nunique() >= 4, "Mancano magnification!"
    return sel, subset, manifest

def build_subset_budget(df, target_gb=1.5, random_state=42, random_within_subtype=False):
    """Config 'diversity': subset a pazienti interi calibrato a ~target_gb (tetto,
    GiB = 1024^3), che MASSIMIZZA i pazienti distinti e protegge i sottotipi rari
    (round-robin tra sottotipi, rari prima; dentro il sottotipo i più leggeri prima).
    Ritorna (selected, subset, manifest)."""
    rng = np.random.default_rng(random_state)
    target_bytes = target_gb * 1024**3
    pats = (df.groupby(['patient_id','label','subtype'], as_index=False)
              .agg(n_images=('filename','count'),
                   bytes=('file_size_bytes','sum')))
    pats['size_mb'] = pats['bytes']/1024**2
    order = (pats.groupby('subtype').patient_id.nunique()
                 .sort_values(kind='mergesort').index.tolist())
    queues = {}
    for st in order:
        g = pats[pats.subtype == st].sample(frac=1.0, random_state=int(rng.integers(0,2**31-1)))
        if not random_within_subtype:
            g = g.sort_values('bytes', kind='mergesort')       # leggeri prima
        queues[st] = g.to_dict('records')
    sel, acc, active = [], 0, True
    while active and acc < target_bytes:
        active = False
        for st in order:
            q = queues[st]
            if not q:
                continue
            r = q[0]
            if sel and acc + r['bytes'] > target_bytes:
                continue
            q.pop(0); sel.append(r['patient_id']); acc += r['bytes']; active = True
    sel = sorted(set(sel))
    subset = df[df.patient_id.isin(sel)].copy()
    manifest = pats[pats.patient_id.isin(sel)].sort_values(['label','subtype','patient_id'])
    assert subset['magnification'].nunique() >= 4, "Mancano magnification!"
    return sel, subset, manifest

# =====================================================================
# SPLIT & K-FOLD
# =====================================================================
def _safe(y):
    y = list(y); return y if (pd.Series(y).value_counts() >= 2).all() else None

def make_splits(subset, random_state=42):
    """image-wise e patient-wise (70/15/15), con verifica anti-leakage."""
    iw = subset.reset_index(drop=True).copy()
    tr, tmp = train_test_split(iw.index, test_size=.30, stratify=_safe(iw.label), random_state=random_state)
    va, te = train_test_split(tmp, test_size=.50, stratify=_safe(iw.loc[tmp,'label']), random_state=random_state)
    iw['split']='train'; iw.loc[va,'split']='val'; iw.loc[te,'split']='test'
    pat = subset[['patient_id','label']].drop_duplicates().reset_index(drop=True)
    ptr, ptmp = train_test_split(pat.patient_id, test_size=.30, stratify=_safe(pat.label), random_state=random_state)
    plbl = pat.set_index('patient_id').loc[ptmp,'label']
    pva, pte = train_test_split(ptmp, test_size=.50, stratify=_safe(plbl), random_state=random_state)
    so = {**{p:'train' for p in ptr}, **{p:'val' for p in pva}, **{p:'test' for p in pte}}
    pw = subset.reset_index(drop=True).copy(); pw['split']=pw.patient_id.map(so)
    S = {s:set(pw.loc[pw.split==s,'patient_id']) for s in ('train','val','test')}
    assert not(S['train']&S['val']) and not(S['train']&S['test']) and not(S['val']&S['test']), "LEAKAGE!"
    return iw, pw

def make_kfold_patient_splits(subset, k=5, val_size=0.15, random_state=42):
    """k-fold PATIENT-WISE stratificata per classe. Ogni paziente è in test una
    sola volta; dentro ogni fold si ritaglia un val (val_size) dai non-test.
    Ritorna un DataFrame long con colonne 'fold' e 'split' (train/val/test)."""
    pat = subset[['patient_id','label']].drop_duplicates().reset_index(drop=True)
    vc = pat['label'].value_counts()
    assert (vc >= k).all(), f"Troppi pochi pazienti per k={k}: {vc.to_dict()}"
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(pat.patient_id, pat.label)):
        test_pat = set(pat.patient_id.iloc[test_idx])
        rem = pat.iloc[train_idx].reset_index(drop=True)
        tr_pat, va_pat = train_test_split(rem.patient_id, test_size=val_size,
            stratify=_safe(rem.label), random_state=random_state+fold_id)
        tr_pat, va_pat = set(tr_pat), set(va_pat)
        assert not (tr_pat & va_pat) and not (tr_pat & test_pat) and not (va_pat & test_pat), \
            f"LEAKAGE nel fold {fold_id}!"
        role = {**{p:'train' for p in tr_pat}, **{p:'val' for p in va_pat}, **{p:'test' for p in test_pat}}
        f = subset.copy(); f['fold'] = fold_id; f['split'] = f['patient_id'].map(role)
        folds.append(f)
    return pd.concat(folds, ignore_index=True)

def verify_folds(kfold, k=5):
    """Verifica: zero overlap pazienti per fold + ogni paziente in test una volta."""
    ok = True
    for fo in range(k):
        s = kfold[kfold.fold == fo]
        S = {sp: set(s.loc[s.split==sp,'patient_id']) for sp in ('train','val','test')}
        if (S['train']&S['val']) or (S['train']&S['test']) or (S['val']&S['test']):
            ok = False
    test_once = (kfold[kfold.split=='test'].groupby('patient_id').fold.nunique() == 1).all()
    return bool(ok and test_once)

# =====================================================================
# STATISTICHE
# =====================================================================
def full_statistics(subset, iw, pw, kfold, dataset_config):
    """Tabella unica (long) con tutte le distribuzioni richieste."""
    rows = []
    def push(view, split, fold, dim, key, sdf):
        rows.append(dict(dataset_config=dataset_config, view=view, split=split, fold=fold,
                         dim=dim, key=str(key),
                         n_images=len(sdf), n_patients=int(sdf.patient_id.nunique())))
    for dim, col in (('class','label'),('subtype','subtype'),('magnification','magnification')):
        for k, s in subset.groupby(col):
            push('subset', None, None, dim, k, s)
    for sp, s in iw.groupby('split'):
        for k, ss in s.groupby('label'):
            push('imagewise', sp, None, 'class', k, ss)
    for sp, s in pw.groupby('split'):
        for k, ss in s.groupby('label'):
            push('patientwise', sp, None, 'class', k, ss)
    for fo, s in kfold.groupby('fold'):
        for sp, ss in s.groupby('split'):
            for k, sss in ss.groupby('label'):
                push('kfold', sp, int(fo), 'class', k, sss)
    return pd.DataFrame(rows)

# =====================================================================
# SALVATAGGIO PER CONFIG
# =====================================================================
def write_config_bundle(subset, manifest, iw, pw, kfold, out_dir, *,
                        dataset_config, selection_strategy, random_state,
                        target_size_gb=None, images_per_subtype_cap=None):
    """Salva l'intera configurazione in out_dir: CSV con colonna dataset_config,
    statistiche, e config.json con i metadati di provenienza."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = lambda df: df.assign(dataset_config=dataset_config)

    tag(subset).to_csv(out_dir/'metadata_subset.csv', index=False)
    tag(manifest).to_csv(out_dir/'subset_manifest.csv', index=False)
    tag(iw).to_csv(out_dir/'image_wise_split.csv', index=False)
    tag(pw).to_csv(out_dir/'patient_wise_split.csv', index=False)
    tag(kfold).to_csv(out_dir/'patient_wise_folds.csv', index=False)
    full_statistics(subset, iw, pw, kfold, dataset_config).to_csv(out_dir/'statistics.csv', index=False)

    b = int(subset['file_size_bytes'].sum())
    pat_lbl = subset[['patient_id','label']].drop_duplicates()
    config = {
        'dataset_config': dataset_config,
        'selection_strategy': selection_strategy,
        'random_state': random_state,
        'target_size_gb': target_size_gb,
        'images_per_subtype_cap': images_per_subtype_cap,
        'n_images': int(len(subset)),
        'n_patients': int(subset['patient_id'].nunique()),
        'n_patients_benign': int((pat_lbl.label=='benign').sum()),
        'n_patients_malignant': int((pat_lbl.label=='malignant').sum()),
        'n_images_benign': int((subset.label=='benign').sum()),
        'n_images_malignant': int((subset.label=='malignant').sum()),
        'actual_size_gib': round(b/1024**3, 3),
        'actual_size_gb_decimal': round(b/1e9, 3),
        'n_subtypes': int(subset['subtype'].nunique()),
        'subtypes': subset['subtype'].value_counts().to_dict(),
        'magnifications': subset['magnification'].value_counts().to_dict(),
        'folds_valid': verify_folds(kfold, k=int(kfold['fold'].nunique())),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    (out_dir/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    return out_dir, config

# =====================================================================
# COMPAT v2/v3 (invariate nel comportamento)
# =====================================================================
def summarize(df_full, subset):
    def blk(df, name):
        out, n = [], len(df)
        for dim, col in (('class','label'),('subtype','subtype'),('magnification','magnification')):
            for k, s in df.groupby(col):
                out.append(dict(split=name, dim=dim, key=str(k), n_patients=s.patient_id.nunique(),
                    n_images=len(s), pct=round(100*len(s)/n,1),
                    size_mb=round(s['file_size_bytes'].sum()/1024**2,1)))
        return pd.DataFrame(out)
    return pd.concat([blk(df_full,'full'), blk(subset,'subset')], ignore_index=True)

def make_experiment_dir(project_root, description):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    d = Path(project_root)/'experiments'/f'{ts}_{description}'
    d.mkdir(parents=True, exist_ok=True)
    return d
