"""
PhysREVE evaluation utilities.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report


@torch.no_grad()
def evaluate(
    model,
    loader:   DataLoader,
    elec_xyz: torch.Tensor,   # (n_ch, 3)
    device:   torch.device,
) -> tuple:
    """
    Run inference on a labeled DataLoader.

    Args:
        model:    PhysREVEFineTuneModel (or any model with the same forward signature)
        loader:   DataLoader yielding (X, y) batches
        elec_xyz: (n_ch, 3) electrode positions on device
        device:   torch.device

    Returns:
        preds:    np.ndarray (n_trials,)  predicted class indices
        labels:   np.ndarray (n_trials,)  ground-truth class indices
        src_acts: torch.Tensor (n_trials, n_src, P)  source activations
    """
    model.eval()
    preds, labels, srcs = [], [], []

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        xyz    = elec_xyz.unsqueeze(0).expand(len(Xb), -1, -1)
        logits, src, _ = model(Xb, xyz)
        preds.append(logits.argmax(1).cpu())
        labels.append(yb.cpu())
        srcs.append(src.cpu())

    return (
        torch.cat(preds).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(srcs),
    )


def print_results(
    preds:       np.ndarray,
    labels:      np.ndarray,
    class_names: list,
    model_name:  str = 'Model',
    chance:      float = None,
):
    """
    Print accuracy, classification report and optionally a comparison to chance.

    Args:
        preds:       predicted class indices
        labels:      ground-truth class indices
        class_names: list of class name strings
        model_name:  label printed in the header
        chance:      chance-level accuracy (e.g. 0.25 for 4-class); if None,
                     computed as 1/n_classes
    """
    n_classes = len(class_names)
    if chance is None:
        chance = 1.0 / n_classes

    acc = (preds == labels).mean()
    print(f'\n{"="*65}')
    print(f'{model_name} — Test accuracy: {acc*100:.1f}%  '
          f'(chance {chance*100:.1f}%,  Δ{(acc-chance)*100:+.1f}%)')
    print(f'{"="*65}')
    print(classification_report(labels, preds, target_names=class_names))
    cm = confusion_matrix(labels, preds)
    print('Confusion matrix:')
    print(cm)
    return acc


def compare_models(
    results: dict,
    class_names: list,
    chance: float = None,
):
    """
    Print a comparison table for multiple models.

    Args:
        results: {model_name: (preds, labels)} dict
        class_names: list of class name strings
        chance: chance level (default: 1/n_classes)
    """
    n_classes = len(class_names)
    if chance is None:
        chance = 1.0 / n_classes

    print(f'\n{"="*65}')
    print(f'{"Model":<40} {"Accuracy":>10} {"vs Chance":>12}')
    print('-' * 65)
    for name, (preds, labels) in results.items():
        acc = (preds == labels).mean()
        print(f'{name:<40} {acc*100:>9.1f}% {(acc-chance)*100:>+11.1f}%')
    print(f'{"Chance level":<40} {chance*100:>9.1f}%')
    print('=' * 65)
