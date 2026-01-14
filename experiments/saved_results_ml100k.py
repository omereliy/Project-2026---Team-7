"""
Saved Experimental Results - ML-100K Dataset
=============================================
This file contains pre-computed results from the ml100k_improved.ipynb notebook.
Import this file to reload results without re-running experiments.

Usage:
    from saved_results_ml100k import load_all_results
    results = load_all_results()

    # Access individual results
    result_bce = results['result_bce']
    robustness_results = results['robustness_results']
    grid_results = results['grid_results']
"""

# ============================================
# PRIMARY EXPERIMENT RESULTS (1:10 Sampling)
# ============================================

# BCE Baseline (1:10 sampling)
RESULT_BCE = {
    'model': 'NeuMF-BCE',
    'best_valid_score': 0.0654,
    'test_result': {
        'hit@5': 0.0669,
        'hit@10': 0.1157,
        'hit@20': 0.1773,
        'ndcg@5': 0.0418,
        'ndcg@10': 0.0575,
        'ndcg@20': 0.0731,
    }
}

# Alpha-Balanced BCE (1:10 sampling, alpha=0.25)
RESULT_ALPHA_BCE = {
    'model': 'NeuMF-AlphaBCE',
    'best_valid_score': 0.0610,
    'alpha': 0.25,
    'test_result': {
        'hit@5': 0.0690,
        'hit@10': 0.1231,
        'hit@20': 0.1953,
        'ndcg@5': 0.0425,
        'ndcg@10': 0.0600,
        'ndcg@20': 0.0780,
    }
}

# Focal Loss (1:10 sampling, gamma=2.0, alpha=0.25)
RESULT_FOCAL = {
    'model': 'NeuMF-FL(g=2.0,a=0.25)',
    'best_valid_score': 0.0628,
    'gamma': 2.0,
    'alpha': 0.25,
    'test_result': {
        'hit@5': 0.0669,
        'hit@10': 0.1062,
        'hit@20': 0.1943,
        'ndcg@5': 0.0428,
        'ndcg@10': 0.0556,
        'ndcg@20': 0.0778,
    }
}

# ============================================
# ROBUSTNESS STUDY RESULTS (Multiple Sampling Ratios)
# ============================================

ROBUSTNESS_RESULTS = {
    'bce': {
        4: {
            'best_valid_score': 0.0524,
            'test_result': {
                'hit@5': 0.0573,
                'hit@10': 0.0998,
                'hit@20': 0.1623,
                'ndcg@5': 0.0330,
                'ndcg@10': 0.0524,
                'ndcg@20': 0.0680,
            }
        },
        10: {
            'best_valid_score': 0.0654,
            'test_result': {
                'hit@5': 0.0669,
                'hit@10': 0.1157,
                'hit@20': 0.1773,
                'ndcg@5': 0.0418,
                'ndcg@10': 0.0575,
                'ndcg@20': 0.0731,
            }
        },
        50: {
            'best_valid_score': 0.0541,
            'test_result': {
                'hit@5': 0.0637,
                'hit@10': 0.1125,
                'hit@20': 0.1847,
                'ndcg@5': 0.0370,
                'ndcg@10': 0.0541,
                'ndcg@20': 0.0720,
            }
        },
    },
    'focal': {
        4: {
            'best_valid_score': 0.0607,
            'test_result': {
                'hit@5': 0.0786,
                'hit@10': 0.1263,
                'hit@20': 0.1996,
                'ndcg@5': 0.0450,
                'ndcg@10': 0.0607,
                'ndcg@20': 0.0790,
            }
        },
        10: {
            'best_valid_score': 0.0628,
            'test_result': {
                'hit@5': 0.0669,
                'hit@10': 0.1062,
                'hit@20': 0.1943,
                'ndcg@5': 0.0428,
                'ndcg@10': 0.0556,
                'ndcg@20': 0.0778,
            }
        },
        50: {
            'best_valid_score': 0.0595,
            'test_result': {
                'hit@5': 0.0765,
                'hit@10': 0.1274,
                'hit@20': 0.2034,
                'ndcg@5': 0.0440,
                'ndcg@10': 0.0595,
                'ndcg@20': 0.0785,
            }
        },
    },
    # Alpha-BCE results (alpha=0.25, gamma=0)
    # NOTE: Update these with actual values when you run the experiments
    # Currently only 1:10 has actual results, others are placeholders marked with TODO
    'alpha_bce': {
        4: {
            'best_valid_score': None,  # TODO: Run experiment to get actual value
            'test_result': {
                'hit@5': None,
                'hit@10': None,
                'hit@20': None,
                'ndcg@5': None,
                'ndcg@10': None,  # TODO: Fill with actual result
                'ndcg@20': None,
            }
        },
        10: {
            'best_valid_score': 0.0610,  # Actual value from experiment
            'test_result': {
                'hit@5': 0.0690,
                'hit@10': 0.1231,
                'hit@20': 0.1953,
                'ndcg@5': 0.0425,
                'ndcg@10': 0.0600,
                'ndcg@20': 0.0780,
            }
        },
        50: {
            'best_valid_score': None,  # TODO: Run experiment to get actual value
            'test_result': {
                'hit@5': None,
                'hit@10': None,
                'hit@20': None,
                'ndcg@5': None,
                'ndcg@10': None,  # TODO: Fill with actual result
                'ndcg@20': None,
            }
        },
    }
}

# ============================================
# GRID SEARCH RESULTS
# 4 gamma values x 3 alpha values x 3 sampling ratios = 36 configs
# ============================================

# Grid search results from ml100k_improved_results.md (actual experimental values)
GRID_RESULTS = [
    # 1:4 sampling (from markdown lines 104-115)
    {'ratio': 4, 'gamma': 0.5, 'alpha': 0.25, 'ndcg@10': 0.0603, 'hit@10': 0.1242},
    {'ratio': 4, 'gamma': 0.5, 'alpha': 0.50, 'ndcg@10': 0.0645, 'hit@10': 0.1274},
    {'ratio': 4, 'gamma': 0.5, 'alpha': 0.75, 'ndcg@10': 0.0657, 'hit@10': 0.1316},  # Best for 1:4
    {'ratio': 4, 'gamma': 1.0, 'alpha': 0.25, 'ndcg@10': 0.0579, 'hit@10': 0.1178},
    {'ratio': 4, 'gamma': 1.0, 'alpha': 0.50, 'ndcg@10': 0.0586, 'hit@10': 0.1125},
    {'ratio': 4, 'gamma': 1.0, 'alpha': 0.75, 'ndcg@10': 0.0617, 'hit@10': 0.1231},
    {'ratio': 4, 'gamma': 2.0, 'alpha': 0.25, 'ndcg@10': 0.0607, 'hit@10': 0.1263},
    {'ratio': 4, 'gamma': 2.0, 'alpha': 0.50, 'ndcg@10': 0.0653, 'hit@10': 0.1253},
    {'ratio': 4, 'gamma': 2.0, 'alpha': 0.75, 'ndcg@10': 0.0612, 'hit@10': 0.1231},
    {'ratio': 4, 'gamma': 3.0, 'alpha': 0.25, 'ndcg@10': 0.0630, 'hit@10': 0.1253},
    {'ratio': 4, 'gamma': 3.0, 'alpha': 0.50, 'ndcg@10': 0.0646, 'hit@10': 0.1242},
    {'ratio': 4, 'gamma': 3.0, 'alpha': 0.75, 'ndcg@10': 0.0525, 'hit@10': 0.1030},  # Worst (unstable)

    # 1:10 sampling (from markdown lines 116-127)
    {'ratio': 10, 'gamma': 0.5, 'alpha': 0.25, 'ndcg@10': 0.0529, 'hit@10': 0.1083},
    {'ratio': 10, 'gamma': 0.5, 'alpha': 0.50, 'ndcg@10': 0.0627, 'hit@10': 0.1221},  # Best for 1:10
    {'ratio': 10, 'gamma': 0.5, 'alpha': 0.75, 'ndcg@10': 0.0533, 'hit@10': 0.1083},
    {'ratio': 10, 'gamma': 1.0, 'alpha': 0.25, 'ndcg@10': 0.0592, 'hit@10': 0.1115},
    {'ratio': 10, 'gamma': 1.0, 'alpha': 0.50, 'ndcg@10': 0.0617, 'hit@10': 0.1168},
    {'ratio': 10, 'gamma': 1.0, 'alpha': 0.75, 'ndcg@10': 0.0515, 'hit@10': 0.1019},
    {'ratio': 10, 'gamma': 2.0, 'alpha': 0.25, 'ndcg@10': 0.0556, 'hit@10': 0.1062},
    {'ratio': 10, 'gamma': 2.0, 'alpha': 0.50, 'ndcg@10': 0.0548, 'hit@10': 0.1051},
    {'ratio': 10, 'gamma': 2.0, 'alpha': 0.75, 'ndcg@10': 0.0441, 'hit@10': 0.0796},  # Very poor
    {'ratio': 10, 'gamma': 3.0, 'alpha': 0.25, 'ndcg@10': 0.0544, 'hit@10': 0.1040},
    {'ratio': 10, 'gamma': 3.0, 'alpha': 0.50, 'ndcg@10': 0.0529, 'hit@10': 0.1030},
    {'ratio': 10, 'gamma': 3.0, 'alpha': 0.75, 'ndcg@10': 0.0533, 'hit@10': 0.1030},

    # 1:50 sampling (from markdown lines 128-139)
    {'ratio': 50, 'gamma': 0.5, 'alpha': 0.25, 'ndcg@10': 0.0524, 'hit@10': 0.1030},
    {'ratio': 50, 'gamma': 0.5, 'alpha': 0.50, 'ndcg@10': 0.0549, 'hit@10': 0.1040},
    {'ratio': 50, 'gamma': 0.5, 'alpha': 0.75, 'ndcg@10': 0.0602, 'hit@10': 0.1178},
    {'ratio': 50, 'gamma': 1.0, 'alpha': 0.25, 'ndcg@10': 0.0555, 'hit@10': 0.1093},
    {'ratio': 50, 'gamma': 1.0, 'alpha': 0.50, 'ndcg@10': 0.0680, 'hit@10': 0.1253},  # Best for 1:50
    {'ratio': 50, 'gamma': 1.0, 'alpha': 0.75, 'ndcg@10': 0.0562, 'hit@10': 0.1093},
    {'ratio': 50, 'gamma': 2.0, 'alpha': 0.25, 'ndcg@10': 0.0595, 'hit@10': 0.1274},
    {'ratio': 50, 'gamma': 2.0, 'alpha': 0.50, 'ndcg@10': 0.0588, 'hit@10': 0.1125},
    {'ratio': 50, 'gamma': 2.0, 'alpha': 0.75, 'ndcg@10': 0.0563, 'hit@10': 0.1062},
    {'ratio': 50, 'gamma': 3.0, 'alpha': 0.25, 'ndcg@10': 0.0598, 'hit@10': 0.1178},
    {'ratio': 50, 'gamma': 3.0, 'alpha': 0.50, 'ndcg@10': 0.0528, 'hit@10': 0.1115},
    {'ratio': 50, 'gamma': 3.0, 'alpha': 0.75, 'ndcg@10': 0.0615, 'hit@10': 0.1263},
]

# ============================================
# HELPER FUNCTION TO LOAD ALL RESULTS
# ============================================

def load_all_results():
    """
    Load all saved experimental results.

    Returns:
        dict: Dictionary containing all results with keys:
            - 'result_bce': BCE baseline results (1:10)
            - 'result_alpha_bce': Alpha-BCE results (1:10)
            - 'result_focal': Focal Loss results (1:10)
            - 'robustness_results': Results across sampling ratios
            - 'grid_results': Grid search results (list of dicts)
    """
    return {
        'result_bce': RESULT_BCE,
        'result_alpha_bce': RESULT_ALPHA_BCE,
        'result_focal': RESULT_FOCAL,
        'robustness_results': ROBUSTNESS_RESULTS,
        'grid_results': GRID_RESULTS,
    }


def print_summary():
    """Print a summary of the saved results."""
    print("="*70)
    print("SAVED RESULTS SUMMARY - ML-100K")
    print("="*70)

    print("\nPrimary Experiment (1:10 sampling):")
    print(f"  BCE:       NDCG@10={RESULT_BCE['test_result']['ndcg@10']:.4f}, HR@10={RESULT_BCE['test_result']['hit@10']:.4f}")
    print(f"  Alpha-BCE: NDCG@10={RESULT_ALPHA_BCE['test_result']['ndcg@10']:.4f}, HR@10={RESULT_ALPHA_BCE['test_result']['hit@10']:.4f}")
    print(f"  Focal:     NDCG@10={RESULT_FOCAL['test_result']['ndcg@10']:.4f}, HR@10={RESULT_FOCAL['test_result']['hit@10']:.4f}")

    print("\nRobustness Study (BCE vs Alpha-BCE vs Focal Loss):")
    print(f"  {'Ratio':<8} {'BCE':<10} {'Alpha-BCE':<12} {'Focal':<10} {'FL vs BCE':<12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    for ratio in [4, 10, 50]:
        bce = ROBUSTNESS_RESULTS['bce'][ratio]['test_result']['ndcg@10']
        fl = ROBUSTNESS_RESULTS['focal'][ratio]['test_result']['ndcg@10']
        alpha = ROBUSTNESS_RESULTS['alpha_bce'][ratio]['test_result']['ndcg@10']

        bce_str = f"{bce:.4f}" if bce else "N/A"
        fl_str = f"{fl:.4f}" if fl else "N/A"
        alpha_str = f"{alpha:.4f}" if alpha else "TODO"

        if bce and fl:
            imp = (fl - bce) / bce * 100
            imp_str = f"{imp:+.1f}%"
        else:
            imp_str = "N/A"

        print(f"  1:{ratio:<6} {bce_str:<10} {alpha_str:<12} {fl_str:<10} {imp_str:<12}")

    # Check for missing Alpha-BCE values
    missing = [r for r in [4, 10, 50] if ROBUSTNESS_RESULTS['alpha_bce'][r]['test_result']['ndcg@10'] is None]
    if missing:
        print(f"\n  NOTE: Alpha-BCE results missing for ratios: {missing}")
        print(f"        Run experiments to complete H3 analysis across all ratios.")

    print(f"\nGrid Search: {len(GRID_RESULTS)} configurations")
    print("="*70)


if __name__ == "__main__":
    print_summary()
