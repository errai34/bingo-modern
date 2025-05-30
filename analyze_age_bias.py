"""
Analyze age underprediction bias in BNN predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def analyze_age_bias(output_dir='./BNN_targeted_output'):
    """Analyze age prediction bias patterns"""

    # Load results
    summary = pd.read_csv(f'{output_dir}/targeted_prediction_summary.csv')
    samples = np.load(f'{output_dir}/targeted_posterior_samples.npz')

    print("="*60)
    print("AGE BIAS ANALYSIS")
    print("="*60)

    # Add age bins for analysis
    age_bins = np.arange(-0.5, 1.5, 0.2)
    summary['age_bin'] = pd.cut(summary['true_age'], age_bins, labels=age_bins[:-1])

    # Calculate bias statistics by age bin
    print("\nBias by Age Bin:")
    print("-"*50)

    bias_stats = []
    for bin_center in age_bins[:-1]:
        mask = (summary['true_age'] >= bin_center) & (summary['true_age'] < bin_center + 0.2)
        if mask.sum() > 0:
            bin_data = summary[mask]
            mean_bias = bin_data['residual'].mean()
            std_bias = bin_data['residual'].std()
            n_stars = len(bin_data)

            bias_stats.append({
                'age_bin': bin_center,
                'mean_bias': mean_bias,
                'std_bias': std_bias,
                'n_stars': n_stars,
                'mean_true': bin_data['true_age'].mean(),
                'mean_pred': bin_data['pred_median'].mean()
            })

            print(f"Age [{bin_center:.1f}, {bin_center+0.2:.1f}): "
                  f"bias={mean_bias:+.3f}±{std_bias:.3f}, n={n_stars}")

    bias_df = pd.DataFrame(bias_stats)

    # Analyze high-age underprediction
    print("\n" + "="*50)
    print("HIGH-AGE ANALYSIS (logAge > 1.0)")
    print("="*50)

    high_age_mask = summary['true_age'] > 1.0
    if high_age_mask.sum() > 0:
        high_age_data = summary[high_age_mask]

        # Calculate systematic bias
        mean_true_high = high_age_data['true_age'].mean()
        mean_pred_high = high_age_data['pred_median'].mean()
        systematic_bias = mean_pred_high - mean_true_high

        print(f"Number of high-age stars: {len(high_age_data)}")
        print(f"Mean true age: {mean_true_high:.3f}")
        print(f"Mean predicted age: {mean_pred_high:.3f}")
        print(f"Systematic bias: {systematic_bias:.3f} dex")
        print(f"Fraction underpredicted: {(high_age_data['residual'] < 0).mean():.1%}")

        # Analyze worst cases
        worst_10pct = high_age_data.nsmallest(int(0.1 * len(high_age_data)), 'residual')
        print(f"\nWorst 10% underpredictions:")
        print(f"Mean true age: {worst_10pct['true_age'].mean():.3f}")
        print(f"Mean predicted age: {worst_10pct['pred_median'].mean():.3f}")
        print(f"Mean bias: {worst_10pct['residual'].mean():.3f}")

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Residuals vs True Age with trend line
    ax = axes[0, 0]
    ax.scatter(summary['true_age'], summary['residual'], alpha=0.5, s=20)

    # Add moving average
    from scipy.ndimage import uniform_filter1d
    sorted_idx = np.argsort(summary['true_age'])
    sorted_ages = summary['true_age'].values[sorted_idx]
    sorted_residuals = summary['residual'].values[sorted_idx]
    window = 50
    if len(sorted_ages) > window:
        smoothed_residuals = uniform_filter1d(sorted_residuals, window, mode='nearest')
        ax.plot(sorted_ages, smoothed_residuals, 'r-', lw=3, label='Moving avg (50 stars)')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('True logAge')
    ax.set_ylabel('Residual (True - Predicted)')
    ax.set_title('Residuals vs True Age')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Bias by age bin
    ax = axes[0, 1]
    ax.bar(bias_df['age_bin'], bias_df['mean_bias'], width=0.15, alpha=0.7, color='coral')
    ax.errorbar(bias_df['age_bin'], bias_df['mean_bias'], yerr=bias_df['std_bias']/np.sqrt(bias_df['n_stars']),
                fmt='none', color='black', capsize=5)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel('Age Bin Center')
    ax.set_ylabel('Mean Bias')
    ax.set_title('Systematic Bias by Age')
    ax.grid(True, alpha=0.3)

    # 3. Prediction vs True with density
    ax = axes[0, 2]
    hexbin = ax.hexbin(summary['true_age'], summary['pred_median'], gridsize=30, cmap='Blues', mincnt=1)
    ax.plot([summary['true_age'].min(), summary['true_age'].max()],
            [summary['true_age'].min(), summary['true_age'].max()], 'r--', lw=2)
    ax.set_xlabel('True logAge')
    ax.set_ylabel('Predicted logAge')
    ax.set_title('Predictions vs True Ages')
    plt.colorbar(hexbin, ax=ax, label='Count')

    # 4. Uncertainty vs Age
    ax = axes[1, 0]
    ax.scatter(summary['true_age'], summary['total_predictive_uncertainty'], alpha=0.5, s=20)
    ax.set_xlabel('True logAge')
    ax.set_ylabel('Total Uncertainty')
    ax.set_title('Predictive Uncertainty vs Age')
    ax.grid(True, alpha=0.3)

    # 5. Normalized residuals distribution by age
    ax = axes[1, 1]
    low_age = summary[summary['true_age'] < 0.5]
    mid_age = summary[(summary['true_age'] >= 0.5) & (summary['true_age'] < 1.0)]
    high_age = summary[summary['true_age'] >= 1.0]

    if len(low_age) > 0:
        ax.hist(low_age['normalized_residual'], bins=30, alpha=0.5, label=f'logAge < 0.5 (n={len(low_age)})', density=True)
    if len(mid_age) > 0:
        ax.hist(mid_age['normalized_residual'], bins=30, alpha=0.5, label=f'0.5 ≤ logAge < 1.0 (n={len(mid_age)})', density=True)
    if len(high_age) > 0:
        ax.hist(high_age['normalized_residual'], bins=30, alpha=0.5, label=f'logAge ≥ 1.0 (n={len(high_age)})', density=True)

    # Add standard normal
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='N(0,1)')
    ax.set_xlabel('Normalized Residual')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Residuals by Age Group')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Feature importance for high-age stars (if we can access training data)
    ax = axes[1, 2]
    # Plot age distribution in training vs test
    ax.hist(summary['true_age'], bins=30, alpha=0.5, label='Test', density=True)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='High age cutoff')
    ax.set_xlabel('logAge')
    ax.set_ylabel('Density')
    ax.set_title('Test Age Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_bias_analysis.png', dpi=300)
    plt.close()

    # Additional analysis: prediction bounds
    print("\n" + "="*50)
    print("PREDICTION RANGE ANALYSIS")
    print("="*50)

    print(f"True age range: [{summary['true_age'].min():.3f}, {summary['true_age'].max():.3f}]")
    print(f"Predicted age range: [{summary['pred_median'].min():.3f}, {summary['pred_median'].max():.3f}]")

    # Check if predictions are compressed
    true_range = summary['true_age'].max() - summary['true_age'].min()
    pred_range = summary['pred_median'].max() - summary['pred_median'].min()
    compression_ratio = pred_range / true_range

    print(f"Range compression ratio: {compression_ratio:.3f}")
    if compression_ratio < 0.9:
        print("WARNING: Predictions are compressed compared to true values!")

    # Analyze prediction saturation
    very_high_age = summary['true_age'] > 1.2
    if very_high_age.sum() > 0:
        print(f"\nVery high age stars (logAge > 1.2):")
        print(f"Number: {very_high_age.sum()}")
        print(f"Max predicted age for these stars: {summary[very_high_age]['pred_median'].max():.3f}")
        print(f"Prediction seems to saturate at: ~{summary['pred_median'].quantile(0.99):.3f}")

    return summary, bias_df

if __name__ == "__main__":
    summary, bias_df = analyze_age_bias()

    # Save bias analysis
    bias_df.to_csv('./BNN_targeted_output/age_bias_statistics.csv', index=False)
    print("\nBias analysis saved to ./BNN_targeted_output/age_bias_statistics.csv")
