"""
PhysREVE — Physics-Informed REVE Pretraining for EEG.

Public API
----------
from physreve import PhysREVEConfig
from physreve.physics import build_leadfield, motor_roi_indices, LeadfieldAttentionBias
from physreve.model import PhysREVEPretrainModel, PhysREVEFineTuneModel
from physreve.losses import pretrain_losses, finetune_losses
from physreve.masking import block_mask
from physreve.data import UnlabeledEEGDataset, LabeledEEGDataset, make_split_loaders
from physreve.train import run_pretraining, run_finetuning, run_baseline_finetune
from physreve.evaluate import evaluate, print_results, compare_models
"""
from .config import PhysREVEConfig
from .physics import build_leadfield, motor_roi_indices, LeadfieldAttentionBias
from .model import PhysREVEPretrainModel, PhysREVEFineTuneModel
from .losses import pretrain_losses, finetune_losses
from .masking import block_mask
from .data import UnlabeledEEGDataset, LabeledEEGDataset, make_split_loaders
from .train import run_pretraining, run_finetuning, run_baseline_finetune
from .evaluate import evaluate, print_results, compare_models

__all__ = [
    'PhysREVEConfig',
    'build_leadfield', 'motor_roi_indices', 'LeadfieldAttentionBias',
    'PhysREVEPretrainModel', 'PhysREVEFineTuneModel',
    'pretrain_losses', 'finetune_losses',
    'block_mask',
    'UnlabeledEEGDataset', 'LabeledEEGDataset', 'make_split_loaders',
    'run_pretraining', 'run_finetuning', 'run_baseline_finetune',
    'evaluate', 'print_results', 'compare_models'
]
