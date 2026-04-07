"""
Simple ML baselines for EEG classification.

Feature extraction (band power + time-domain) + LDA / Logistic Regression /
XGBoost classifiers. Used for ablation tables to anchor deep-learning results
against classical methods.

Usage
-----
    from physreve.baselines import extract_features, run_ml_baselines

    X_tr, y_tr = ...   # (N, C, T) numpy arrays, labels (N,)
    X_val, y_val = ...

    results = run_ml_baselines(X_tr, y_tr, X_val, y_val, sfreq=250)
    # {'lda': 0.62, 'logreg': 0.65, 'xgboost': 0.68}
"""
import numpy as np

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False


# ── Feature extraction ───────────────────────────────────────────────────────

def _bandpower(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """
    Mean power in [lo, hi] Hz band.

    x: (..., T)
    Returns: (...,) — same leading dims, one scalar per channel/trial
    """
    fft   = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)
    mask  = (freqs >= lo) & (freqs <= hi)
    return np.abs(fft[..., mask]).mean(axis=-1)


def extract_features(
    X:     np.ndarray,   # (N, C, T) z-scored EEG
    sfreq: float = 250.0,
    bands: list  = None,
) -> np.ndarray:
    """
    Extract per-channel band-power and time-domain features.

    Bands (Hz) — default: delta, theta, alpha, beta, low-gamma:
        (1,4), (4,8), (8,13), (13,30), (30,50)

    Time-domain per channel: variance, kurtosis (4th moment).

    Returns
    -------
    feats: (N, C * (n_bands + 2))   float32
        All features are stacked channel-major, then flattened.
    """
    if bands is None:
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]

    N, C, T = X.shape
    parts   = []

    for lo, hi in bands:
        bp = _bandpower(X, sfreq, lo, hi)   # (N, C)
        parts.append(bp)

    # Variance
    parts.append(X.var(axis=-1))            # (N, C)
    # Excess kurtosis
    mu  = X.mean(axis=-1, keepdims=True)
    std = X.std( axis=-1, keepdims=True).clip(min=1e-8)
    parts.append((((X - mu) / std) ** 4 - 3).mean(axis=-1))  # (N, C)

    feats = np.stack(parts, axis=2)   # (N, C, n_features)
    return feats.reshape(N, -1).astype(np.float32)


# ── Individual classifiers ───────────────────────────────────────────────────

def run_lda(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> float:
    """
    LDA with shrinkage ('auto').

    Returns val accuracy.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lda',    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ])
    clf.fit(X_tr, y_tr)
    return float((clf.predict(X_val) == y_val).mean())


def run_logreg(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    C: float = 0.1,
    max_iter: int = 1000,
) -> float:
    """
    L2 logistic regression.

    Returns val accuracy.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(C=C, max_iter=max_iter)),
    ])
    clf.fit(X_tr, y_tr)
    return float((clf.predict(X_val) == y_val).mean())


def run_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_estimators: int = 300,
    max_depth:    int = 4,
    lr:           float = 0.05,
) -> float:
    """
    XGBoost classifier.

    Returns val accuracy.
    """
    if not _XGB:
        raise ImportError("xgboost required: pip install xgboost")
    n_classes = len(np.unique(y_tr))
    obj = 'multi:softmax' if n_classes > 2 else 'binary:logistic'
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        objective=obj,
        num_class=n_classes if n_classes > 2 else None,
        use_label_encoder=False,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return float((clf.predict(X_val) == y_val).mean())


# ── Convenience wrapper ──────────────────────────────────────────────────────

def run_ml_baselines(
    X_tr:   np.ndarray,   # (N_tr, C, T) z-scored EEG
    y_tr:   np.ndarray,   # (N_tr,)
    X_val:  np.ndarray,   # (N_val, C, T)
    y_val:  np.ndarray,   # (N_val,)
    sfreq:  float = 250.0,
    bands:  list  = None,
    models: list  = None,  # subset of ['lda', 'logreg', 'xgboost']
) -> dict:
    """
    Run all simple ML baselines and return a dict of val accuracies.

    Args:
        X_tr, y_tr:   training set — numpy arrays (N, C, T) and (N,)
        X_val, y_val: validation set
        sfreq:        sampling frequency for band-power extraction
        bands:        list of (lo, hi) Hz tuples (default: delta–low-gamma)
        models:       which models to run (default: all available)

    Returns:
        dict mapping model name → val accuracy, e.g.
        {'lda': 0.62, 'logreg': 0.65, 'xgboost': 0.68}
    """
    if models is None:
        models = ['lda', 'logreg']
        if _XGB:
            models.append('xgboost')

    print('Extracting EEG features ...')
    feats_tr  = extract_features(X_tr,  sfreq=sfreq, bands=bands)
    feats_val = extract_features(X_val, sfreq=sfreq, bands=bands)
    print(f'  Feature matrix: train {feats_tr.shape}, val {feats_val.shape}')

    runners = {
        'lda':     run_lda,
        'logreg':  run_logreg,
        'xgboost': run_xgboost,
    }

    results = {}
    for name in models:
        if name not in runners:
            print(f'  [skip] unknown model: {name}')
            continue
        try:
            acc = runners[name](feats_tr, y_tr, feats_val, y_val)
            results[name] = acc
            print(f'  {name:12s}  val_acc = {acc:.3f}')
        except ImportError as e:
            print(f'  {name:12s}  [skip] {e}')
        except Exception as e:
            print(f'  {name:12s}  [error] {e}')

    return results
