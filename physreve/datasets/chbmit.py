"""
CHB-MIT Scalp EEG Database loader.
https://physionet.org/content/chbmit/1.0.0/

Dataset notes
─────────────
- Bipolar montage: channel names like "FP1-F7"
- 256 Hz sampling rate
- 23 channels (standard 10-20 bipolar derivations)
- Open access — no account required

For PhysREVE, we:
  1. Resample to target_sfreq (default 200 Hz) to match BCI IV 2a config
  2. Build a bipolar leadfield: L_bip[i] = L_ref[e1] − L_ref[e2]
  3. Use electrode midpoints for 4D positional encoding
  4. Binary classification: interictal (0) vs ictal (1)
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
import requests
from tqdm.auto import tqdm

from ..physics import build_leadfield
from ..data import UnlabeledEEGDataset, LabeledEEGDataset, make_split_loaders

# ── Electrode name normalisation: CHB-MIT uppercase → MNE standard_1020 ──────
_NORM = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ':  'Fz',  'CZ':  'Cz',  'PZ':  'Pz',
    # Old 10-20 names used by some CHB-MIT files
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    # These are already correct in MNE standard_1020
    'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
    'C3': 'C3', 'C4': 'C4',
    'P3': 'P3', 'P4': 'P4',
    'O1': 'O1', 'O2': 'O2',
    'T7': 'T7', 'T8': 'T8',
    'P7': 'P7', 'P8': 'P8',
    'FT9': 'FT9', 'FT10': 'FT10',
}


def _norm_elec(name: str) -> str:
    name = name.strip().upper()
    return _NORM.get(name, name.capitalize())


def _parse_bipolar_pair(ch_name: str):
    """'FP1-F7' → ('Fp1', 'F7').  Single electrode → ('Fp1', None)."""
    ch_name = ch_name.strip()
    # Remove trailing suffixes like '-0', '-1' from CHB-MIT duplicate channels
    ch_name = re.sub(r'-\d+$', '', ch_name)
    if '-' in ch_name:
        parts = ch_name.split('-', 1)
        return _norm_elec(parts[0]), _norm_elec(parts[1])
    return _norm_elec(ch_name), None


# ── Download ──────────────────────────────────────────────────────────────────

BASE_URL = 'https://physionet.org/files/chbmit/1.0.0'


def _download_file(url: str, dest: Path, desc: str = None):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True,
        desc=desc or dest.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def download_patient(patient_id: str = 'chb01', data_dir: str = './data/chb-mit',
                     files: list = None):
    """
    Download CHB-MIT EDF files and summary for one patient.

    Args:
        patient_id: e.g. 'chb01'
        data_dir:   local directory to cache files
        files:      list of EDF filenames to download (without path).
                    If None, downloads only the summary file.
    """
    patient_dir = Path(data_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Summary file (always)
    summary_name = f'{patient_id}-summary.txt'
    _download_file(
        f'{BASE_URL}/{patient_id}/{summary_name}',
        patient_dir / summary_name,
        desc=summary_name,
    )
    print(f'Summary downloaded: {patient_dir / summary_name}')

    if files:
        for fname in files:
            _download_file(
                f'{BASE_URL}/{patient_id}/{fname}',
                patient_dir / fname,
                desc=fname,
            )
        print(f'Downloaded {len(files)} EDF file(s) to {patient_dir}/')


# ── Summary / annotation parsing ─────────────────────────────────────────────

def parse_summary(summary_path: str) -> dict:
    """
    Parse a CHB-MIT summary file.

    Returns
    -------
    dict mapping filename → list of (start_sec, end_sec) tuples.
    Files with no seizures have an empty list.
    """
    result = {}
    current_file = None
    n_seizures = 0
    seizure_count = 0

    with open(summary_path) as f:
        for line in f:
            line = line.strip()

            m = re.match(r'File Name:\s+(\S+)', line)
            if m:
                current_file = m.group(1)
                result[current_file] = []
                n_seizures = 0
                seizure_count = 0
                continue

            m = re.match(r'Number of Seizures in File:\s+(\d+)', line)
            if m:
                n_seizures = int(m.group(1))
                continue

            m = re.match(r'Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)\s+seconds', line)
            if m and current_file:
                start = int(m.group(1))
                result[current_file].append([start, None])
                seizure_count += 1
                continue

            m = re.match(r'Seizure(?:\s+\d+)?\s+End Time:\s+(\d+)\s+seconds', line)
            if m and current_file and result[current_file]:
                result[current_file][-1][1] = int(m.group(1))
                continue

    # Convert to tuple pairs
    return {k: [(s, e) for s, e in v if e is not None]
            for k, v in result.items()}


# ── EEG loading & windowing ───────────────────────────────────────────────────

def _is_eeg_channel(ch_name: str) -> bool:
    """Heuristic: reject ECG, EMG, VNS, DC channels."""
    upper = ch_name.upper()
    bad_prefixes = ('ECG', 'EMG', 'VNS', 'DC', '--', 'LOC', 'ROC', 'EKG')
    return not any(upper.startswith(p) for p in bad_prefixes)


def load_patient_epochs(
    data_dir:           str,
    patient_id:         str,
    seizure_info:       dict,
    window_sec:         float = 4.0,
    stride_sec:         float = 2.0,
    target_sfreq:       float = 200.0,
    ictal_buffer_sec:   float = 2.0,   # exclude n sec at seizure onset/offset
    interictal_gap_sec: float = 60.0,  # min gap from seizure for interictal
    max_interictal_per_file: int = 200,
    verbose:            bool  = False,
):
    """
    Load CHB-MIT EDF files and extract ictal and interictal windows.

    Args:
        data_dir:           path containing patient_id/ subdirectory
        patient_id:         e.g. 'chb01'
        seizure_info:       output of parse_summary()
        window_sec:         epoch duration in seconds
        stride_sec:         stride between epochs in seconds
        target_sfreq:       resample to this frequency
        ictal_buffer_sec:   exclude this many seconds at seizure boundaries
        interictal_gap_sec: interictal windows must be this far from any seizure
        max_interictal_per_file: cap interictal windows per file (balance)
        verbose:            print per-file stats

    Returns:
        ictal_X:       np.ndarray (n_ictal, n_ch, n_times)
        interictal_X:  np.ndarray (n_interictal, n_ch, n_times)
        ch_names:      list of channel names from the EDF
    """
    import mne

    patient_dir = Path(data_dir) / patient_id
    window_smp  = int(window_sec   * target_sfreq)
    stride_smp  = int(stride_sec   * target_sfreq)

    ictal_X, interictal_X = [], []
    ch_names_out = None

    edf_files = sorted(patient_dir.glob('*.edf'))
    if not edf_files:
        raise FileNotFoundError(f'No EDF files found in {patient_dir}')

    for edf_path in edf_files:
        fname = edf_path.name
        seizures = seizure_info.get(fname, [])

        # Load raw EEG
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        # Select EEG channels only
        eeg_chs = [ch for ch in raw.ch_names if _is_eeg_channel(ch)]
        raw.pick_channels(eeg_chs)

        # Resample
        if abs(raw.info['sfreq'] - target_sfreq) > 1:
            raw.resample(target_sfreq, verbose=False)

        data = raw.get_data()  # (n_ch, n_times)
        n_ch, n_times = data.shape
        sfreq = raw.info['sfreq']

        if ch_names_out is None:
            ch_names_out = list(raw.ch_names)

        # ── Extract ictal windows ─────────────────────────────────────────
        for onset_s, offset_s in seizures:
            buf = ictal_buffer_sec
            ictal_start = int((onset_s  + buf) * sfreq)
            ictal_end   = int((offset_s - buf) * sfreq)

            pos = ictal_start
            while pos + window_smp <= ictal_end:
                ictal_X.append(data[:, pos:pos + window_smp])
                pos += stride_smp

        # ── Extract interictal windows ────────────────────────────────────
        # Build a boolean mask: True = too close to a seizure
        near_seizure = np.zeros(n_times, dtype=bool)
        gap_smp = int(interictal_gap_sec * sfreq)
        for onset_s, offset_s in seizures:
            lo = max(0, int(onset_s  * sfreq) - gap_smp)
            hi = min(n_times, int(offset_s * sfreq) + gap_smp)
            near_seizure[lo:hi] = True

        inter_windows = []
        pos = 0
        while pos + window_smp <= n_times:
            if not near_seizure[pos:pos + window_smp].any():
                inter_windows.append(data[:, pos:pos + window_smp])
            pos += stride_smp

        # Cap to avoid overwhelming the dataset with interictal
        if inter_windows:
            rng = np.random.default_rng(42)
            n_keep = min(len(inter_windows), max_interictal_per_file)
            chosen = rng.choice(len(inter_windows), n_keep, replace=False)
            interictal_X.extend([inter_windows[i] for i in chosen])

        if verbose:
            n_ictal = sum(1 for _ in seizures)
            print(f'  {fname}: {n_ictal} seizure(s), '
                  f'{len(ictal_X)} ictal / {len(interictal_X)} interictal windows total')

    ictal_X      = np.stack(ictal_X,      axis=0).astype(np.float32) if ictal_X      else np.empty((0, n_ch, window_smp), np.float32)
    interictal_X = np.stack(interictal_X, axis=0).astype(np.float32) if interictal_X else np.empty((0, n_ch, window_smp), np.float32)

    print(f'\nData extracted:')
    print(f'  Ictal (seizure)     : {len(ictal_X):>5} windows  '
          f'× ({n_ch} ch, {window_smp} samples)')
    print(f'  Interictal (normal) : {len(interictal_X):>5} windows')

    return ictal_X, interictal_X, ch_names_out


# ── Bipolar leadfield ─────────────────────────────────────────────────────────

def build_leadfield_bipolar(
    bipolar_ch_names: list,
    sfreq:    float = 200.0,
    grid_pos: float = 10.0,
    mindist:  float = 5.0,
    verbose:  bool  = False,
):
    """
    Compute the EEG leadfield matrix for bipolar channel derivations.

    For each bipolar pair "E1-E2":
        L_bip[i, :] = L_ref[idx(E1), :] − L_ref[idx(E2), :]

    This is the physically correct forward model: the potential difference
    between E1 and E2 is the difference of their individual scalp potentials.

    Returns
    -------
    L_col    : np.ndarray  (n_bip, n_src)   column-normalised
    L_row    : np.ndarray  (n_bip, n_src)   row-normalised
    src_pos  : np.ndarray  (n_src, 3)       source positions (metres)
    info_ref : mne.Info                     info for unique electrodes
    elec_xyz : np.ndarray  (n_bip, 3)       midpoint positions per bipolar pair
    valid_mask: list[bool]                  True = pair was successfully built
    """
    # Parse bipolar pairs
    pairs = [_parse_bipolar_pair(ch) for ch in bipolar_ch_names]

    # Unique electrode names present in at least one pair
    unique = list({e for p in pairs for e in p if e is not None})

    if verbose:
        print(f'  Unique electrodes: {sorted(unique)}')

    # Build referential leadfield for unique electrodes
    _, _, src_pos, info_ref, L_ref = build_leadfield(
        unique, sfreq=sfreq, grid_pos=grid_pos, mindist=mindist,
        verbose=verbose, return_raw=True
    )
    # L_ref: (n_unique, n_src)

    # Map electrode name → row index in L_ref
    ref_ch_names = [info_ref['chs'][i]['ch_name'] for i in range(len(info_ref['chs']))]
    elec_to_idx  = {name: i for i, name in enumerate(ref_ch_names)}
    elec_pos     = np.stack([info_ref['chs'][i]['loc'][:3] for i in range(len(info_ref['chs']))])

    # Build bipolar leadfield rows
    L_bip_rows = []
    midpoints   = []
    valid_mask  = []

    for e1, e2 in pairs:
        i1 = elec_to_idx.get(e1)
        i2 = elec_to_idx.get(e2) if e2 is not None else None

        if i1 is None:
            valid_mask.append(False)
            L_bip_rows.append(np.zeros(L_ref.shape[1], np.float32))
            midpoints.append(np.zeros(3, np.float32))
            continue

        if i2 is None:
            row = L_ref[i1]
            mid = elec_pos[i1]
        else:
            row = L_ref[i1] - L_ref[i2]
            mid = (elec_pos[i1] + elec_pos[i2]) / 2.0

        L_bip_rows.append(row.astype(np.float32))
        midpoints.append(mid.astype(np.float32))
        valid_mask.append(True)

    L_bip = np.stack(L_bip_rows)  # (n_bip, n_src)

    # Column-normalise (per source)
    col_n = np.linalg.norm(L_bip, axis=0, keepdims=True).clip(min=1e-8)
    L_col = L_bip / col_n

    # Row-normalise (per channel)
    row_n = np.linalg.norm(L_bip, axis=1, keepdims=True).clip(min=1e-8)
    L_row = L_bip / row_n

    elec_xyz = np.stack(midpoints)  # (n_bip, 3)

    if verbose:
        n_valid = sum(valid_mask)
        print(f'  Bipolar leadfield: {n_valid}/{len(pairs)} channels mapped')
        print(f'  L_bip shape: {L_bip.shape}')

    return L_col, L_row, src_pos, info_ref, elec_xyz, valid_mask


# ── Convenience: build balanced datasets ─────────────────────────────────────

def make_seizure_loaders(
    ictal_X:       np.ndarray,   # (n_ictal, n_ch, n_times)
    interictal_X:  np.ndarray,   # (n_interictal, n_ch, n_times)
    train_frac:    float = 0.70,
    val_frac:      float = 0.15,
    batch_size:    int   = 16,
    seed:          int   = 42,
    balance:       bool  = True,  # undersample interictal to match ictal count
):
    """
    Build train/val/test loaders for binary seizure detection.

    Labels: 0 = interictal, 1 = ictal (seizure).

    When balance=True, interictal is downsampled to match ictal count so that
    the model doesn't trivially predict the majority class.

    Returns:
        pretrain_loader: DataLoader of interictal-only (unlabeled) for pretraining
        train_loader, val_loader, test_loader: labeled loaders
    """
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(seed)

    if balance and len(interictal_X) > len(ictal_X):
        idx = rng.choice(len(interictal_X), len(ictal_X), replace=False)
        inter_balanced = interictal_X[idx]
    else:
        inter_balanced = interictal_X

    # Labeled dataset: ictal + balanced interictal
    X = np.concatenate([ictal_X, inter_balanced])
    y = np.concatenate([
        np.ones(len(ictal_X), dtype=np.int64),
        np.zeros(len(inter_balanced), dtype=np.int64),
    ])

    train_loader, val_loader, test_loader = make_split_loaders(
        X, y, train_frac=train_frac, val_frac=val_frac,
        batch_size=batch_size, seed=seed,
    )

    # Pretraining: use ALL interictal data (unlabeled)
    pretrain_ds = UnlabeledEEGDataset(interictal_X)
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    print(f'\nLoaders built:')
    print(f'  Pretrain (interictal, unlabeled): {len(pretrain_ds)} windows')
    print(f'  Train : {len(train_loader.dataset)} | '
          f'Val: {len(val_loader.dataset)} | '
          f'Test: {len(test_loader.dataset)}')
    print(f'  Class balance — ictal: {len(ictal_X)}, interictal: {len(inter_balanced)}')

    return pretrain_loader, train_loader, val_loader, test_loader
