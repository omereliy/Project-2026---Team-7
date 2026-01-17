#!/usr/bin/env python3
"""
Per-User Wilcoxon Signed-Rank Test for BCE vs Focal Loss
=========================================================

This script extracts per-user metrics from RecBole models and runs
a proper Wilcoxon signed-rank test with n=943 paired observations
(one per user in ML-100K).

Usage:
    Run this in Colab after training BCE and FL models, or after
    the cells that train the models in ml100k_improved.ipynb.

    # After training models, add this cell:
    from per_user_wilcoxon import run_per_user_wilcoxon_analysis
    results = run_per_user_wilcoxon_analysis(
        bce_result=result_bce,
        fl_result=result_focal,
        test_data=test_data
    )
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def get_per_user_metrics(model, trainer, test_data, config, k_list=[5, 10, 20]):
    """
    Extract per-user NDCG and Hit Rate metrics from a trained model.

    Args:
        model: Trained RecBole model
        trainer: RecBole Trainer object
        test_data: Test DataLoader
        config: RecBole config
        k_list: List of k values for @k metrics

    Returns:
        dict: Per-user metrics with user_ids as keys
    """
    from recbole.evaluator import Evaluator
    from recbole.utils import get_gpu_usage
    import torch

    model.eval()
    evaluator = Evaluator(config)

    # Collect predictions and ground truth per user
    user_metrics = {}

    with torch.no_grad():
        for batch_idx, batched_data in enumerate(test_data):
            interaction, history_index, positive_u, positive_i = batched_data

            # Get model scores
            scores = model.full_sort_predict(interaction)

            # Get user IDs from this batch
            user_ids = interaction['user_id'].cpu().numpy()

            # For each user in batch, compute metrics
            for i, user_id in enumerate(user_ids):
                if user_id in user_metrics:
                    continue  # Already processed this user

                # Get this user's scores and ground truth
                user_scores = scores[i].cpu().numpy()

                # Get positive items for this user
                pos_items = positive_i[positive_u == i].cpu().numpy()

                if len(pos_items) == 0:
                    continue

                # Compute ranking metrics for this user
                user_metrics[user_id] = compute_user_metrics(
                    user_scores, pos_items, k_list
                )

    return user_metrics


def compute_user_metrics(scores, pos_items, k_list):
    """
    Compute NDCG and Hit Rate for a single user.

    Args:
        scores: Item scores from model (array of shape [n_items])
        pos_items: Ground truth positive item indices
        k_list: List of k values

    Returns:
        dict: Metrics for this user
    """
    # Get ranked item indices (highest score first)
    ranked_items = np.argsort(-scores)

    metrics = {}
    for k in k_list:
        top_k = ranked_items[:k]

        # Hit Rate@k: 1 if any positive item in top-k, else 0
        hits = np.isin(top_k, pos_items).astype(float)
        hit_rate = 1.0 if hits.sum() > 0 else 0.0

        # NDCG@k
        dcg = 0.0
        for rank, item in enumerate(top_k):
            if item in pos_items:
                dcg += 1.0 / np.log2(rank + 2)  # rank is 0-indexed

        # Ideal DCG (all positives at top)
        n_pos = min(len(pos_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_pos))

        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics[f'hit@{k}'] = hit_rate
        metrics[f'ndcg@{k}'] = ndcg

    return metrics


def extract_per_user_from_evaluator(trainer, test_data):
    """
    Alternative: Extract per-user metrics using RecBole's internal evaluator.

    This hooks into RecBole's evaluation pipeline to get per-user scores
    before they are aggregated.
    """
    from recbole.evaluator import Evaluator, Collector
    import torch

    model = trainer.model
    config = trainer.config

    model.eval()

    # Create collector to gather results
    if hasattr(trainer, 'eval_collector'):
        collector = trainer.eval_collector
    else:
        collector = Collector(config)

    collector.data_collect(test_data)

    # Get per-user metrics from collector's struct
    # This depends on RecBole version
    try:
        struct = collector.get_data_struct()
        # RecBole stores per-user results in the struct
        return struct
    except Exception as e:
        print(f"Could not extract from collector: {e}")
        return None


def run_wilcoxon_test(bce_user_metrics, fl_user_metrics, metric='ndcg@10'):
    """
    Run Wilcoxon signed-rank test on per-user metrics.

    Args:
        bce_user_metrics: dict of user_id -> metrics for BCE model
        fl_user_metrics: dict of user_id -> metrics for FL model
        metric: Which metric to test (e.g., 'ndcg@10', 'hit@10')

    Returns:
        dict: Test results
    """
    # Get common users
    common_users = set(bce_user_metrics.keys()) & set(fl_user_metrics.keys())
    n_users = len(common_users)

    print(f"Number of paired observations: {n_users}")

    # Extract paired metrics
    bce_scores = []
    fl_scores = []

    for user_id in sorted(common_users):
        bce_scores.append(bce_user_metrics[user_id][metric])
        fl_scores.append(fl_user_metrics[user_id][metric])

    bce_scores = np.array(bce_scores)
    fl_scores = np.array(fl_scores)
    differences = fl_scores - bce_scores

    # Basic stats
    mean_bce = np.mean(bce_scores)
    mean_fl = np.mean(fl_scores)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    # Count wins
    fl_wins = np.sum(differences > 0)
    bce_wins = np.sum(differences < 0)
    ties = np.sum(differences == 0)

    # Wilcoxon signed-rank test (two-sided)
    stat_two, p_two = stats.wilcoxon(differences, alternative='two-sided')

    # Wilcoxon signed-rank test (one-sided: FL > BCE)
    stat_greater, p_greater = stats.wilcoxon(differences, alternative='greater')

    # Effect size: rank-biserial correlation
    # r = 1 - (2*W) / (n*(n+1)/2) where W is the smaller of W+ and W-
    n_nonzero = np.sum(differences != 0)
    r_effect = 1 - (2 * stat_two) / (n_nonzero * (n_nonzero + 1) / 2)

    # Cohen's d for paired samples
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    return {
        'n_users': n_users,
        'mean_bce': mean_bce,
        'mean_fl': mean_fl,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'fl_wins': fl_wins,
        'bce_wins': bce_wins,
        'ties': ties,
        'wilcoxon_stat': stat_two,
        'p_value_two_sided': p_two,
        'p_value_one_sided': p_greater,
        'rank_biserial_r': r_effect,
        'cohens_d': cohens_d,
    }


def interpret_results(results, metric='ndcg@10', alpha=0.05):
    """Print interpretation of Wilcoxon test results."""
    print("=" * 70)
    print(f"WILCOXON SIGNED-RANK TEST RESULTS ({metric.upper()})")
    print("=" * 70)

    print(f"\nSample Size: n = {results['n_users']} users")

    print(f"\nDescriptive Statistics:")
    print(f"  BCE mean:  {results['mean_bce']:.4f}")
    print(f"  FL mean:   {results['mean_fl']:.4f}")
    print(f"  Mean diff: {results['mean_diff']:+.4f} ({results['mean_diff']/results['mean_bce']*100:+.1f}%)")
    print(f"  Std diff:  {results['std_diff']:.4f}")

    print(f"\nWin/Loss/Tie:")
    print(f"  FL wins:  {results['fl_wins']} ({results['fl_wins']/results['n_users']*100:.1f}%)")
    print(f"  BCE wins: {results['bce_wins']} ({results['bce_wins']/results['n_users']*100:.1f}%)")
    print(f"  Ties:     {results['ties']} ({results['ties']/results['n_users']*100:.1f}%)")

    print(f"\nStatistical Tests:")
    print(f"  Wilcoxon statistic: {results['wilcoxon_stat']:.2f}")
    print(f"  p-value (two-sided): {results['p_value_two_sided']:.6f}")
    print(f"  p-value (one-sided, FL > BCE): {results['p_value_one_sided']:.6f}")

    sig_two = "YES" if results['p_value_two_sided'] < alpha else "NO"
    sig_one = "YES" if results['p_value_one_sided'] < alpha else "NO"
    print(f"  Significant at α={alpha} (two-sided): {sig_two}")
    print(f"  Significant at α={alpha} (one-sided): {sig_one}")

    print(f"\nEffect Sizes:")
    print(f"  Cohen's d: {results['cohens_d']:.3f}", end="")
    d = abs(results['cohens_d'])
    if d < 0.2:
        print(" (negligible)")
    elif d < 0.5:
        print(" (small)")
    elif d < 0.8:
        print(" (medium)")
    else:
        print(" (large)")

    print(f"  Rank-biserial r: {results['rank_biserial_r']:.3f}", end="")
    r = abs(results['rank_biserial_r'])
    if r < 0.1:
        print(" (negligible)")
    elif r < 0.3:
        print(" (small)")
    elif r < 0.5:
        print(" (medium)")
    else:
        print(" (large)")

    print("\n" + "=" * 70)

    # Conclusion
    if results['p_value_one_sided'] < alpha and results['mean_diff'] > 0:
        print("CONCLUSION: Focal Loss significantly outperforms BCE")
    elif results['p_value_one_sided'] < alpha and results['mean_diff'] < 0:
        print("CONCLUSION: BCE significantly outperforms Focal Loss")
    else:
        print("CONCLUSION: No significant difference between FL and BCE")

    print("=" * 70)


def run_per_user_wilcoxon_analysis(bce_result, fl_result, test_data,
                                    metrics=['ndcg@10', 'hit@10']):
    """
    Main function to run per-user Wilcoxon analysis.

    Args:
        bce_result: Result dict from training BCE model (contains 'model', 'trainer')
        fl_result: Result dict from training FL model (contains 'model', 'trainer')
        test_data: Test DataLoader
        metrics: List of metrics to analyze

    Returns:
        dict: Results for each metric
    """
    print("Extracting per-user metrics for BCE model...")
    bce_user_metrics = get_per_user_metrics(
        bce_result['model'],
        bce_result['trainer'],
        test_data,
        bce_result['trainer'].config
    )

    print("Extracting per-user metrics for FL model...")
    fl_user_metrics = get_per_user_metrics(
        fl_result['model'],
        fl_result['trainer'],
        test_data,
        fl_result['trainer'].config
    )

    all_results = {}

    for metric in metrics:
        print(f"\n{'='*70}")
        print(f"Analyzing {metric.upper()}")
        print(f"{'='*70}")

        results = run_wilcoxon_test(bce_user_metrics, fl_user_metrics, metric)
        interpret_results(results, metric)
        all_results[metric] = results

    return all_results


# =============================================================================
# SIMPLIFIED VERSION: Use when you have aggregate results only
# This simulates what a per-user test might look like based on aggregate stats
# =============================================================================

def simulate_per_user_test_from_aggregates(
    bce_mean, fl_mean, n_users=943,
    assumed_std_ratio=0.5, metric_name='ndcg@10'
):
    """
    Estimate what a per-user test might show based on aggregate results.

    This is an approximation that assumes:
    - Per-user metrics are normally distributed
    - Standard deviation is approximately std_ratio * mean

    Args:
        bce_mean: Aggregate BCE metric
        fl_mean: Aggregate FL metric
        n_users: Number of users
        assumed_std_ratio: Assumed coefficient of variation
        metric_name: Name of metric for display

    Returns:
        dict: Estimated test results
    """
    print("=" * 70)
    print(f"ESTIMATED PER-USER ANALYSIS (from aggregate stats)")
    print("=" * 70)
    print("\nWARNING: This is an estimate. For accurate results, run the")
    print("full per-user analysis with run_per_user_wilcoxon_analysis()")
    print("-" * 70)

    # Estimate per-user standard deviations
    bce_std = bce_mean * assumed_std_ratio
    fl_std = fl_mean * assumed_std_ratio

    # Simulate per-user scores
    np.random.seed(42)
    bce_scores = np.random.normal(bce_mean, bce_std, n_users)
    fl_scores = np.random.normal(fl_mean, fl_std, n_users)

    # Clip to valid range
    bce_scores = np.clip(bce_scores, 0, 1)
    fl_scores = np.clip(fl_scores, 0, 1)

    differences = fl_scores - bce_scores

    # Run Wilcoxon test
    stat, p_value = stats.wilcoxon(differences, alternative='greater')

    # Effect size
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    fl_wins = np.sum(differences > 0)

    print(f"\nMetric: {metric_name}")
    print(f"BCE mean: {bce_mean:.4f}, FL mean: {fl_mean:.4f}")
    print(f"Mean difference: {mean_diff:+.4f} ({mean_diff/bce_mean*100:+.1f}%)")
    print(f"\nSimulated with n={n_users} users:")
    print(f"  FL wins: {fl_wins}/{n_users} ({fl_wins/n_users*100:.1f}%)")
    print(f"  Wilcoxon p-value (one-sided): {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    sig = "YES" if p_value < 0.05 else "NO"
    print(f"  Significant at α=0.05: {sig}")

    return {
        'p_value': p_value,
        'cohens_d': cohens_d,
        'fl_wins': fl_wins,
        'n_users': n_users
    }


if __name__ == "__main__":
    # Demo with aggregate results from saved_results_ml100k.py
    from saved_results_ml100k import ROBUSTNESS_RESULTS

    print("\n" + "="*70)
    print("PER-USER WILCOXON TEST ESTIMATION")
    print("(Based on aggregate results - for accurate test, re-run models)")
    print("="*70)

    for ratio in [4, 10, 50]:
        bce_ndcg = ROBUSTNESS_RESULTS['bce'][ratio]['test_result']['ndcg@10']
        fl_ndcg = ROBUSTNESS_RESULTS['focal'][ratio]['test_result']['ndcg@10']

        print(f"\n{'='*70}")
        print(f"SAMPLING RATIO 1:{ratio}")
        print(f"{'='*70}")

        simulate_per_user_test_from_aggregates(
            bce_ndcg, fl_ndcg,
            n_users=943,
            metric_name=f'ndcg@10 (1:{ratio})'
        )
