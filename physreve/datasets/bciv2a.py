"""
BCI IV 2a (BNCI2014_001) data loader via MOABB.

Channel layout and class mapping match the PhysREVE notebook.
"""
import numpy as np
from torch.utils.data import DataLoader

from ..config import PhysREVEConfig
from ..data import UnlabeledEEGDataset, LabeledEEGDataset, make_split_loaders

CH_NAMES = [
    'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1',  'Pz',  'P2',  'POz'
]
N_CLASSES   = 4
SFREQ       = 200.0
CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
LABEL_MAP   = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}


def load_pretrain_data(
    cfg:              PhysREVEConfig,
    pretrain_subjects: list = None,
    batch_size:        int  = 16,
) -> tuple:
    """
    Load BCI IV 2a pretraining data (default: subjects 2-9, no labels).

    Returns:
        pretrain_loader: DataLoader of UnlabeledEEGDataset
        T_win:           int — number of time samples per trial
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    if pretrain_subjects is None:
        pretrain_subjects = list(range(2, 10))

    dataset  = BNCI2014_001()
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand', 'feet', 'tongue'],
        n_classes=4, fmin=0.5, fmax=40.0, tmin=0.5, tmax=2.5
    )

    print(f'Loading pretraining subjects {pretrain_subjects}...')
    parts = []
    for subj in pretrain_subjects:
        X_s, _, _ = paradigm.get_data(dataset, subjects=[subj], return_epochs=False)
        P_s = X_s.shape[-1] // cfg.patch_size
        X_s = X_s[:, :, :P_s * cfg.patch_size].astype(np.float32)
        parts.append(X_s)
        print(f'  Subject {subj}: {X_s.shape[0]} trials')

    X = np.concatenate(parts, axis=0)
    T_win = X.shape[-1]
    print(f'Pretraining pool: {X.shape[0]} trials')

    loader = DataLoader(
        UnlabeledEEGDataset(X),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    return loader, T_win


def load_finetune_data(
    cfg:             PhysREVEConfig,
    finetune_subject: int   = 1,
    train_frac:       float = 0.70,
    val_frac:         float = 0.15,
    batch_size:       int   = 16,
    seed:             int   = 42,
) -> tuple:
    """
    Load BCI IV 2a fine-tuning data for one subject.

    Returns:
        train_loader, val_loader, test_loader, T_win
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset  = BNCI2014_001()
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand', 'feet', 'tongue'],
        n_classes=4, fmin=0.5, fmax=40.0, tmin=0.5, tmax=2.5
    )

    print(f'Loading fine-tuning subject {finetune_subject}...')
    X, y_str, _ = paradigm.get_data(dataset, subjects=[finetune_subject], return_epochs=False)
    P   = X.shape[-1] // cfg.patch_size
    X   = X[:, :, :P * cfg.patch_size].astype(np.float32)
    y   = np.array([LABEL_MAP[yi] for yi in y_str])
    T_win = X.shape[-1]
    print(f'  Subject {finetune_subject}: {X.shape[0]} trials  ({P} patches per trial)')

    train_loader, val_loader, test_loader = make_split_loaders(
        X, y, train_frac=train_frac, val_frac=val_frac,
        batch_size=batch_size, seed=seed
    )
    return train_loader, val_loader, test_loader, T_win
