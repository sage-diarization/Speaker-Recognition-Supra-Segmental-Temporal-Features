"""
Compare reproduced experiment results against a paper's reported results,
using only summary statistics (mean, std, n) for each side.

Supports a nested structure of experiments -> tests -> (yours, paper) stats,
and reports:
  - a per-test comparison (Welch's t-test, Cohen's d, CI on the difference)
  - a per-experiment comparison, pooling that experiment's tests together
  - an overall comparison, pooling everything together

Usage:
    Fill in the DATA dict in the `if __name__ == "__main__":` block below
    with your experiments/tests, then run the script. `compare()` and
    `pool_stats()` can also be imported and used directly.
"""

import numpy as np
from scipy.stats import ttest_ind_from_stats, t as t_dist


def compare(your_mean, your_std, your_n,
            paper_mean, paper_std, paper_n,
            alpha=0.05, metric_name="metric"):
    """
    Compare two sets of summary statistics (e.g. your reproduction vs. a paper)
    using Welch's t-test, Cohen's d, and a confidence interval on the difference.

    Returns a dict with all computed values; also prints a readable report.
    """

    # --- Welch's t-test (unequal variances assumed) ---
    t_stat, p_value = ttest_ind_from_stats(
        mean1=your_mean, std1=your_std, nobs1=your_n,
        mean2=paper_mean, std2=paper_std, nobs2=paper_n,
        equal_var=False,
    )

    # --- Effect size: Cohen's d (pooled std) ---
    pooled_std = np.sqrt((your_std**2 + paper_std**2) / 2)
    cohens_d = (your_mean - paper_mean) / pooled_std if pooled_std > 0 else float("nan")

    # --- Confidence interval on the difference (Welch-Satterthwaite df) ---
    se_diff = np.sqrt(your_std**2 / your_n + paper_std**2 / paper_n)
    diff = your_mean - paper_mean

    df_num = (your_std**2 / your_n + paper_std**2 / paper_n) ** 2
    df_den = (
        (your_std**2 / your_n) ** 2 / (your_n - 1)
        + (paper_std**2 / paper_n) ** 2 / (paper_n - 1)
    )
    df = df_num / df_den if df_den > 0 else float("nan")

    t_crit = t_dist.ppf(1 - alpha / 2, df)
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    # --- Effect size interpretation (rule of thumb) ---
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        d_label = "negligible"
    elif abs_d < 0.5:
        d_label = "small"
    elif abs_d < 0.8:
        d_label = "medium"
    else:
        d_label = "large"

    significant = p_value < alpha

    # --- Report ---
    print(f"=== Comparison for '{metric_name}' ===")
    print(f"Yours:  mean={your_mean:.4f}  std={your_std:.4f}  n={your_n}")
    print(f"Paper:  mean={paper_mean:.4f}  std={paper_std:.4f}  n={paper_n}")
    print(f"Difference (yours - paper): {diff:.4f}")
    print(f"95% CI on difference: [{ci_low:.4f}, {ci_high:.4f}]  (df={df:.1f})")
    print(f"Welch's t-test: t={t_stat:.3f}, p={p_value:.4f} "
          f"({'significant' if significant else 'not significant'} at alpha={alpha})")
    print(f"Cohen's d: {cohens_d:.3f} ({d_label} effect)")
    print()

    if not significant and abs_d < 0.5:
        verdict = "Consistent with a successful reproduction."
    elif significant and abs_d >= 0.5:
        verdict = ("Meaningful discrepancy detected — worth investigating "
                   "implementation, hyperparameters, or data differences.")
    elif significant and abs_d < 0.5:
        verdict = ("Statistically significant but small effect size — "
                   "likely not practically meaningful, but note it.")
    else:
        verdict = ("Not statistically significant, but effect size is non-trivial — "
                   "inconclusive; more runs would help.")
    print(f"Verdict: {verdict}")

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_label": d_label,
        "diff": diff,
        "ci": (ci_low, ci_high),
        "df": df,
        "significant": significant,
        "verdict": verdict,
    }


def pool_stats(stats_list):
    """
    Combine several (mean, std, n) groups into one pooled (mean, std, n).

    Uses the standard formula for combining sample statistics across groups,
    accounting for both within-group variance and between-group variance
    (i.e. it correctly inflates the pooled std if the group means differ,
    rather than just averaging the stds).

    stats_list: list of (mean, std, n) tuples.
    """
    means = np.array([s[0] for s in stats_list], dtype=float)
    stds = np.array([s[1] for s in stats_list], dtype=float)
    ns = np.array([s[2] for s in stats_list], dtype=float)

    n_total = ns.sum()
    mean_total = (ns * means).sum() / n_total

    within = ((ns - 1) * stds**2).sum()
    between = (ns * (means - mean_total) ** 2).sum()
    var_total = (within + between) / (n_total - 1)

    return mean_total, np.sqrt(var_total), int(n_total)


def run_comparisons(data, alpha=0.05):
    """
    data: nested dict of the form
        {
            "Experiment name": {
                "Test name": {
                    "yours": (mean, std, n),
                    "paper": (mean, std, n),
                },
                ...
            },
            ...
        }

    Prints per-test, per-experiment (pooled), and overall (pooled) comparisons.
    Returns a dict mirroring the same structure, with comparison results
    plus "__experiment__" and top-level "__overall__" entries.
    """
    results = {}
    all_yours, all_paper = [], []

    for exp_name, tests in data.items():
        print("#" * 70)
        print(f"# EXPERIMENT: {exp_name}")
        print("#" * 70)
        print()

        results[exp_name] = {}
        exp_yours, exp_paper = [], []

        for test_name, stats in tests.items():
            your_mean, your_std, your_n = stats["yours"]
            paper_mean, paper_std, paper_n = stats["paper"]

            results[exp_name][test_name] = compare(
                your_mean, your_std, your_n,
                paper_mean, paper_std, paper_n,
                alpha=alpha,
                metric_name=f"{exp_name} / {test_name}",
            )

            exp_yours.append((your_mean, your_std, your_n))
            exp_paper.append((paper_mean, paper_std, paper_n))

        # Pooled comparison across this experiment's tests
        pooled_your = pool_stats(exp_yours)
        pooled_paper = pool_stats(exp_paper)

        print("-" * 70)
        print(f"Pooled result for experiment '{exp_name}' "
              f"({len(exp_yours)} tests combined)")
        print("-" * 70)
        results[exp_name]["__experiment__"] = compare(
            *pooled_your, *pooled_paper,
            alpha=alpha,
            metric_name=f"{exp_name} (pooled across tests)",
        )

        all_yours.extend(exp_yours)
        all_paper.extend(exp_paper)

    # Pooled comparison across all experiments/tests
    pooled_your_overall = pool_stats(all_yours)
    pooled_paper_overall = pool_stats(all_paper)

    print("=" * 70)
    print(f"OVERALL pooled result ({len(all_yours)} tests across "
          f"{len(data)} experiments combined)")
    print("=" * 70)
    results["__overall__"] = compare(
        *pooled_your_overall, *pooled_paper_overall,
        alpha=alpha,
        metric_name="overall (pooled across all experiments)",
    )

    return results


if __name__ == "__main__":
    # --- Edit this with your actual experiments, tests, and stats ---
    # Each (mean, std, n) is computed from your 5 runs per test, and the
    # paper's reported (mean, std, n) for that same test.

    run_comparisons({
        "CNN-timit-OS": {
            "OS": {
                "yours": (0.0, 0.0, 5),
                "paper": (0.0, 0.0, 5),
            },
            "SS": {
                "yours": (12.75, 3.48209706929603, 5),
                "paper": (9.00, 2.15, 5),
            },
        },
        "CNN-timit-SS": {
            "OS": {
                "yours": (12.0, 9.766012492312305, 5),
                "paper": (9.0, 1.66, 5),
            },
            "SS": {
                "yours": (0.0, 0.0, 5),
                "paper": (1.25, 0.0, 5),
            },
        },
    })

    # NOTE: pooling assumes the tests within an experiment (or across
    # experiments) measure a comparable quantity (e.g. the same metric on
    # related conditions). If the tests are measuring genuinely different
    # things, pooling them into one number can be misleading — inspect the
    # per-test results above before leaning on the pooled/overall verdict.

    print("=" * 70)

    run_comparisons({
        "RNN-timit-OS": {
            "OS": {
                "yours": (4.5, 1.274754878398196, 5),
                "paper": (1.25, 1.12, 5),
            },
            "SS": {
                "yours": (5, 1.369306393762915, 5),
                "paper": (2.75, 0.5, 5),
            },
        },
        "RNN-timit-SS": {
            "OS": {
                "yours": (5.75, 1.6955824957813166, 5),
                "paper": (2.0, 1.0, 5),
            },
            "SS": {
                "yours": (3.75, 1.118033988749895, 5),
                "paper": (0.25, 0.5, 5),
            },
        },
    })

    run_comparisons({
        "ResNet-timit-OS": {
            "OS": {
                "yours": (3.5, 1.5, 5),
                "paper": (1.0, 0.94, 5),
            },
            "SS": {
                "yours": (11.25, 2.7386127875258306, 5),
                "paper": (12.75, 4.430011286667337, 5),
            },
        },
        "ResNet-timit-SS": {
            "OS": {
                "yours": (4.25, 3.020761493398643, 5),
                "paper": (2.75, 0.94, 5),
            },
            "SS": {
                "yours": (4.75, 2.2912878474779195, 5),
                "paper": (1.0, 0.94, 5),
            },
        },
    })