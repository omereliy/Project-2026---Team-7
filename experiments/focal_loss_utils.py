"""
Focal Loss Utilities for NCF Experiments
=========================================
Shared utilities for NCF with Focal Loss experiments on MovieLens datasets.

This module implements:
1. Focal Loss and Alpha-Balanced BCE loss functions
2. Custom NeuMF models with different loss functions
3. Training dynamics tracking for mechanism validation
4. Experiment configuration presets
5. Evaluation and comparison utilities

Reference: "Addressing Class Imbalance in NCF with Focal Loss" (AAMAS 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple, Any

# RecBole imports (lazy import to avoid issues when module is imported)
def get_recbole_imports():
    """Lazy import of RecBole modules."""
    from recbole.model.general_recommender.neumf import NeuMF
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import init_seed, init_logger
    return NeuMF, Config, create_dataset, data_preparation, Trainer, init_seed, init_logger


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in recommendation systems.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        gamma (float): Focusing parameter. Higher values down-weight easy examples more.
                      gamma=0 reduces to alpha-balanced BCE. Default: 2.0
        alpha (float): Class balancing weight for positive class. Default: 0.25
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted probabilities (after sigmoid), shape [batch_size]
            targets: Ground truth labels (0 or 1), shape [batch_size]

        Returns:
            Focal loss value
        """
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)

        # Calculate p_t (probability of true class)
        p_t = targets * inputs + (1 - targets) * (1 - inputs)

        # Calculate alpha_t (class weight)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)
        focal_loss = -focal_weight * torch.log(p_t)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def get_sample_weights(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get per-sample focal weights for training dynamics analysis."""
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return alpha_t * torch.pow(1 - p_t, self.gamma)


class AlphaBalancedBCE(nn.Module):
    """
    Alpha-Balanced BCE Loss (Focal Loss with gamma=0).

    This serves as a control to isolate the focusing effect from class weighting.

    Loss = -alpha_t * log(p_t)

    Args:
        alpha (float): Class balancing weight for positive class. Default: 0.25
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, alpha: float = 0.25, reduction: str = 'mean'):
        super(AlphaBalancedBCE, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = -alpha_t * torch.log(p_t)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# TRAINING DYNAMICS TRACKER
# =============================================================================

class TrainingDynamicsTracker:
    """
    Tracks training dynamics to validate Focal Loss mechanism.

    Tracks:
    1. Loss contribution by confidence bin
    2. Gradient magnitude from easy vs hard examples
    3. Sample difficulty evolution over epochs
    """

    def __init__(self, confidence_bins: List[float] = None):
        self.confidence_bins = confidence_bins or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.history = {
            'epoch': [],
            'loss_by_bin': [],
            'count_by_bin': [],
            'grad_norm_easy': [],
            'grad_norm_hard': [],
            'mean_confidence_pos': [],
            'mean_confidence_neg': [],
        }

    def compute_loss_by_confidence_bin(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """Compute loss contribution from each confidence bin."""
        predictions = predictions.detach()
        targets = targets.detach()

        # Compute p_t for binning
        p_t = targets * predictions + (1 - targets) * (1 - predictions)

        bin_losses = {}
        bin_counts = {}

        for i in range(len(self.confidence_bins) - 1):
            low, high = self.confidence_bins[i], self.confidence_bins[i+1]
            mask = (p_t >= low) & (p_t < high)

            bin_name = f"[{low:.1f},{high:.1f})"
            if mask.sum() > 0:
                # Compute loss for this bin
                with torch.enable_grad():
                    pred_bin = predictions[mask].clone().requires_grad_(True)
                    tgt_bin = targets[mask]
                    bin_loss = loss_fn(pred_bin, tgt_bin)
                bin_losses[bin_name] = bin_loss.item()
                bin_counts[bin_name] = mask.sum().item()
            else:
                bin_losses[bin_name] = 0.0
                bin_counts[bin_name] = 0

        return bin_losses, bin_counts

    def record_epoch(
        self,
        epoch: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
        model: nn.Module = None
    ):
        """Record metrics for one epoch."""
        self.history['epoch'].append(epoch)

        # Loss by confidence bin
        loss_by_bin, count_by_bin = self.compute_loss_by_confidence_bin(
            predictions, targets, loss_fn
        )
        self.history['loss_by_bin'].append(loss_by_bin)
        self.history['count_by_bin'].append(count_by_bin)

        # Mean confidence for positives and negatives
        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() > 0:
            self.history['mean_confidence_pos'].append(predictions[pos_mask].mean().item())
        else:
            self.history['mean_confidence_pos'].append(0.0)

        if neg_mask.sum() > 0:
            self.history['mean_confidence_neg'].append((1 - predictions[neg_mask]).mean().item())
        else:
            self.history['mean_confidence_neg'].append(0.0)

    def get_summary_df(self) -> pd.DataFrame:
        """Get summary as DataFrame."""
        records = []
        for i, epoch in enumerate(self.history['epoch']):
            record = {'epoch': epoch}
            if i < len(self.history['loss_by_bin']):
                for bin_name, loss in self.history['loss_by_bin'][i].items():
                    record[f'loss_{bin_name}'] = loss
                    record[f'count_{bin_name}'] = self.history['count_by_bin'][i].get(bin_name, 0)
            if i < len(self.history['mean_confidence_pos']):
                record['mean_conf_pos'] = self.history['mean_confidence_pos'][i]
                record['mean_conf_neg'] = self.history['mean_confidence_neg'][i]
            records.append(record)
        return pd.DataFrame(records)


# =============================================================================
# CUSTOM NEUMF MODELS
# =============================================================================

def create_neumf_focal_loss_class():
    """Factory function to create NeuMF_FocalLoss class with RecBole imports."""
    NeuMF, *_ = get_recbole_imports()

    class NeuMF_FocalLoss(NeuMF):
        """NeuMF model with Focal Loss instead of BCE."""

        def __init__(self, config, dataset, gamma=2.0, alpha=0.25, track_dynamics=False):
            super(NeuMF_FocalLoss, self).__init__(config, dataset)
            self.gamma = gamma
            self.alpha = alpha
            self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction='mean')
            self.track_dynamics = track_dynamics
            self.dynamics_tracker = TrainingDynamicsTracker() if track_dynamics else None
            self._epoch_predictions = []
            self._epoch_targets = []
            print(f"Initialized NeuMF with Focal Loss (gamma={gamma}, alpha={alpha})")

        def calculate_loss(self, interaction):
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
            label = interaction[self.LABEL]
            output = self.forward(user, item)
            loss = self.focal_loss(output, label)

            # Track for dynamics analysis
            if self.track_dynamics:
                self._epoch_predictions.append(output.detach())
                self._epoch_targets.append(label.detach())

            return loss

        def record_epoch_dynamics(self, epoch: int):
            """Call at end of epoch to record dynamics."""
            if self.track_dynamics and self._epoch_predictions:
                all_preds = torch.cat(self._epoch_predictions)
                all_targets = torch.cat(self._epoch_targets)
                self.dynamics_tracker.record_epoch(
                    epoch, all_preds, all_targets, self.focal_loss
                )
                self._epoch_predictions = []
                self._epoch_targets = []

    return NeuMF_FocalLoss


def create_neumf_alpha_bce_class():
    """Factory function to create NeuMF_AlphaBCE class with RecBole imports."""
    NeuMF, *_ = get_recbole_imports()

    class NeuMF_AlphaBCE(NeuMF):
        """NeuMF model with Alpha-Balanced BCE (gamma=0 control)."""

        def __init__(self, config, dataset, alpha=0.25, track_dynamics=False):
            super(NeuMF_AlphaBCE, self).__init__(config, dataset)
            self.alpha = alpha
            self.alpha_bce = AlphaBalancedBCE(alpha=alpha, reduction='mean')
            self.track_dynamics = track_dynamics
            self.dynamics_tracker = TrainingDynamicsTracker() if track_dynamics else None
            self._epoch_predictions = []
            self._epoch_targets = []
            print(f"Initialized NeuMF with Alpha-Balanced BCE (alpha={alpha})")

        def calculate_loss(self, interaction):
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
            label = interaction[self.LABEL]
            output = self.forward(user, item)
            loss = self.alpha_bce(output, label)

            if self.track_dynamics:
                self._epoch_predictions.append(output.detach())
                self._epoch_targets.append(label.detach())

            return loss

    return NeuMF_AlphaBCE


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

def get_base_config(dataset: str, device: torch.device, neg_sample_num: int = 4) -> Dict:
    """
    Get base configuration for experiments.

    Args:
        dataset: 'ml-100k' or 'ml-1m'
        device: torch device
        neg_sample_num: Number of negative samples per positive (4, 10, or 50)

    Returns:
        Configuration dictionary
    """
    return {
        # Dataset
        'dataset': dataset,
        'data_path': './dataset/',

        # Data preprocessing
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'threshold': {'rating': 4},
        'val_interval': {'rating': '[4,inf)'},

        # Evaluation settings
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full',
        },

        # Training negative sampling
        'train_neg_sample_args': {
            'distribution': 'uniform',
            'sample_num': neg_sample_num,
            'dynamic': False,
        },

        # Evaluation settings
        'metrics': ['Hit', 'NDCG'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',

        # Training settings
        'epochs': 100,
        'stopping_step': 10,
        'train_batch_size': 256,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,

        # Reproducibility
        'seed': 42,

        # Device
        'device': device,

        # Logging
        'show_progress': True,
    }


def get_neumf_config(base_config: Dict) -> Dict:
    """Add NeuMF-specific configuration."""
    config = base_config.copy()
    config.update({
        'model': 'NeuMF',
        'mf_embedding_size': 64,
        'mlp_embedding_size': 64,
        'mlp_hidden_size': [128, 64, 32],
        'dropout_prob': 0.0,
    })
    return config


def compute_effective_class_ratio(alpha: float, neg_sample_ratio: int) -> float:
    """
    Compute effective class weight ratio.

    Effective Ratio = ((1-alpha) * neg_ratio) / alpha

    Args:
        alpha: Class balancing weight for positives
        neg_sample_ratio: Number of negatives per positive

    Returns:
        Effective ratio of negative to positive weight
    """
    if alpha == 0:
        return float('inf')
    return ((1 - alpha) * neg_sample_ratio) / alpha


def get_balanced_alpha(neg_sample_ratio: int) -> float:
    """
    Compute alpha value that balances effective class weights.

    For effective ratio = 1: alpha = neg_ratio / (neg_ratio + 1)

    Args:
        neg_sample_ratio: Number of negatives per positive

    Returns:
        Alpha value for balanced weighting
    """
    return neg_sample_ratio / (neg_sample_ratio + 1)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_neumf_focal_loss(
    config_dict: Dict,
    dataset: str,
    gamma: float = 2.0,
    alpha: float = 0.25,
    seed: int = 42,
    track_dynamics: bool = False
) -> Dict:
    """
    Train NeuMF with Focal Loss.

    Args:
        config_dict: Configuration dictionary
        dataset: Dataset name ('ml-100k' or 'ml-1m')
        gamma: Focal Loss focusing parameter
        alpha: Focal Loss class balancing weight
        seed: Random seed
        track_dynamics: Whether to track training dynamics

    Returns:
        Dictionary with training results
    """
    NeuMF, Config, create_dataset, data_preparation, Trainer, init_seed, init_logger = get_recbole_imports()
    NeuMF_FocalLoss = create_neumf_focal_loss_class()

    init_seed(seed, reproducibility=True)
    config = Config(model='NeuMF', dataset=dataset, config_dict=config_dict)
    init_logger(config)
    logger = logging.getLogger()

    dataset_obj = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    model = NeuMF_FocalLoss(
        config, dataset_obj, gamma=gamma, alpha=alpha, track_dynamics=track_dynamics
    ).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    result = {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'model': model,
        'trainer': trainer,
        'gamma': gamma,
        'alpha': alpha,
    }

    if track_dynamics and model.dynamics_tracker:
        result['dynamics'] = model.dynamics_tracker.get_summary_df()

    return result


def train_neumf_alpha_bce(
    config_dict: Dict,
    dataset: str,
    alpha: float = 0.25,
    seed: int = 42,
    track_dynamics: bool = False
) -> Dict:
    """
    Train NeuMF with Alpha-Balanced BCE (gamma=0 control).

    Args:
        config_dict: Configuration dictionary
        dataset: Dataset name
        alpha: Class balancing weight
        seed: Random seed
        track_dynamics: Whether to track training dynamics

    Returns:
        Dictionary with training results
    """
    NeuMF, Config, create_dataset, data_preparation, Trainer, init_seed, init_logger = get_recbole_imports()
    NeuMF_AlphaBCE = create_neumf_alpha_bce_class()

    init_seed(seed, reproducibility=True)
    config = Config(model='NeuMF', dataset=dataset, config_dict=config_dict)
    init_logger(config)
    logger = logging.getLogger()

    dataset_obj = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    model = NeuMF_AlphaBCE(
        config, dataset_obj, alpha=alpha, track_dynamics=track_dynamics
    ).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    result = {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'model': model,
        'trainer': trainer,
        'alpha': alpha,
    }

    if track_dynamics and model.dynamics_tracker:
        result['dynamics'] = model.dynamics_tracker.get_summary_df()

    return result


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def create_comparison_table(results_list: List[Dict], model_names: List[str]) -> pd.DataFrame:
    """
    Create comparison table for multiple models.

    Args:
        results_list: List of result dictionaries
        model_names: List of model names

    Returns:
        DataFrame with comparison
    """
    metrics = ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']

    data = []
    for metric in metrics:
        row = {'Metric': metric.upper()}
        for name, result in zip(model_names, results_list):
            val = result['test_result'].get(metric, 0)
            row[name] = f'{val:.4f}'
        data.append(row)

    return pd.DataFrame(data)


def compute_improvement(baseline_results: Dict, comparison_results: Dict) -> Dict:
    """Compute improvement of comparison over baseline."""
    metrics = ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
    improvements = {}

    for metric in metrics:
        baseline = baseline_results['test_result'].get(metric, 0)
        comparison = comparison_results['test_result'].get(metric, 0)

        if baseline > 0:
            pct = (comparison - baseline) / baseline * 100
        else:
            pct = 0

        improvements[metric] = {
            'baseline': baseline,
            'comparison': comparison,
            'diff': comparison - baseline,
            'pct_change': pct
        }

    return improvements


def validate_focal_loss_implementation():
    """Test that Focal Loss implementation is correct."""
    bce_loss = nn.BCELoss()

    preds = torch.tensor([0.9, 0.1, 0.5, 0.8])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    bce = bce_loss(preds, targets)

    # Test 1: FL(gamma=0, alpha=0.5) = 0.5 * BCE
    focal_loss_gamma0 = FocalLoss(gamma=0.0, alpha=0.5)
    fl_alpha05 = focal_loss_gamma0(preds, targets)

    assert abs(fl_alpha05.item() - 0.5 * bce.item()) < 0.01, \
        "FL(gamma=0, alpha=0.5) should equal 0.5*BCE"

    # Test 2: Higher gamma reduces loss for well-classified examples
    focal_loss_gamma2 = FocalLoss(gamma=2.0, alpha=0.5)
    fl_gamma2 = focal_loss_gamma2(preds, targets)

    assert fl_gamma2.item() < fl_alpha05.item(), \
        "Higher gamma should reduce loss"

    print("Focal Loss implementation PASSED all tests!")
    return True


def demonstrate_focal_loss_effect():
    """Demonstrate how Focal Loss down-weights easy examples."""
    bce_loss = nn.BCELoss(reduction='none')
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25, reduction='none')

    scenarios = [
        ("Easy negative (pred=0.05, y=0)", torch.tensor([0.05]), torch.tensor([0.0])),
        ("Hard positive (pred=0.3, y=1)", torch.tensor([0.30]), torch.tensor([1.0])),
        ("Hard negative (pred=0.7, y=0)", torch.tensor([0.70]), torch.tensor([0.0])),
        ("Easy positive (pred=0.95, y=1)", torch.tensor([0.95]), torch.tensor([1.0])),
    ]

    results = []
    for desc, pred, target in scenarios:
        bce = bce_loss(pred, target).item()
        fl = focal_loss(pred, target).item()
        ratio = bce / fl if fl > 0 else float('inf')
        results.append({
            'scenario': desc,
            'bce_loss': bce,
            'focal_loss': fl,
            'reduction_factor': ratio
        })

    return pd.DataFrame(results)


# =============================================================================
# ALPHA-SAMPLING INTERACTION ANALYSIS
# =============================================================================

def analyze_alpha_sampling_interaction(
    neg_ratios: List[int] = [4, 10, 50],
    alphas: List[float] = [0.25, 0.5, 0.75]
) -> pd.DataFrame:
    """
    Analyze the interaction between alpha and negative sampling ratio.

    Returns DataFrame showing effective class weight ratios.
    """
    records = []
    for neg_ratio in neg_ratios:
        balanced_alpha = get_balanced_alpha(neg_ratio)
        for alpha in alphas:
            eff_ratio = compute_effective_class_ratio(alpha, neg_ratio)
            records.append({
                'neg_ratio': f'1:{neg_ratio}',
                'alpha': alpha,
                'effective_ratio': f'{eff_ratio:.1f}:1',
                'balanced_alpha': f'{balanced_alpha:.2f}',
                'is_balanced': abs(eff_ratio - 1.0) < 0.1
            })

    return pd.DataFrame(records)


# =============================================================================
# MULTI-SEED EXPERIMENT RUNNER
# =============================================================================

def run_multi_seed_experiment(
    config_dict: Dict,
    dataset: str,
    loss_type: str,  # 'bce', 'alpha_bce', or 'focal'
    gamma: float = 2.0,
    alpha: float = 0.25,
    seeds: List[int] = None,
    track_dynamics: bool = False
) -> Dict:
    """
    Run experiment with multiple seeds for statistical testing.

    Args:
        config_dict: Configuration dictionary
        dataset: Dataset name
        loss_type: 'bce', 'alpha_bce', or 'focal'
        gamma: Focal Loss gamma (only for 'focal')
        alpha: Class balancing weight
        seeds: List of random seeds
        track_dynamics: Whether to track training dynamics

    Returns:
        Dictionary with aggregated results
    """
    from recbole.quick_start import run_recbole

    if seeds is None:
        seeds = list(range(10))

    all_results = []

    for seed in seeds:
        config_dict['seed'] = seed

        if loss_type == 'bce':
            config_dict['loss_type'] = 'BCE'
            result = run_recbole(model='NeuMF', dataset=dataset, config_dict=config_dict)
            all_results.append({
                'seed': seed,
                'test_result': result['test_result'],
                'best_valid_score': result['best_valid_score']
            })
        elif loss_type == 'alpha_bce':
            result = train_neumf_alpha_bce(
                config_dict, dataset, alpha=alpha, seed=seed, track_dynamics=track_dynamics
            )
            all_results.append({
                'seed': seed,
                'test_result': result['test_result'],
                'best_valid_score': result['best_valid_score']
            })
        elif loss_type == 'focal':
            result = train_neumf_focal_loss(
                config_dict, dataset, gamma=gamma, alpha=alpha, seed=seed,
                track_dynamics=track_dynamics
            )
            all_results.append({
                'seed': seed,
                'test_result': result['test_result'],
                'best_valid_score': result['best_valid_score']
            })

    # Aggregate metrics
    metrics = ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
    aggregated = {}

    for metric in metrics:
        values = [r['test_result'].get(metric, 0) for r in all_results]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    return {
        'all_results': all_results,
        'aggregated': aggregated,
        'loss_type': loss_type,
        'gamma': gamma if loss_type == 'focal' else None,
        'alpha': alpha,
        'seeds': seeds
    }
