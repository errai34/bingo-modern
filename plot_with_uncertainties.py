"""
Plot predicted vs true ages with model uncertainties
Shows error bars and uncertainty trends
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_predictions_with_uncertainties(summary_path='./BNN_targeted_output/targeted_prediction_summary.csv',
                                      samples_path='./BNN_targeted_output/targeted_posterior_samples.npz'):
    """Create comprehensive plots showing predictions with uncertainties"""

    # Load data
    print("Loading prediction summary...")
    summary = pd.read_csv(summary_path)

    print("Loading posterior samples...")
    samples_data = np.load(samples_path)
    total_samples = samples_data['total_samples']
    mean_predictions = samples_data['mean_predictions']

    # Calculate percentiles for each star
    pred_16 = np.percentile(total_samples, 16, axis=0)
    pred_84 = np.percentile(total_samples, 84, axis=0)
    pred_2_5 = np.percentile(total_samples, 2.5, axis=0)
    pred_97_5 = np.percentile(total_samples, 97.5, axis=0)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ========== Plot 1: Predicted vs True with Error Bars ==========
    ax1 = axes[0, 0]

    # Sort by true age for cleaner visualization
    sort_idx = np.argsort(summary['true_age'])

    # Plot with error bars (68% CI)
    ax1.errorbar(summary['true_age'][sort_idx],
                summary['pred_median'][sort_idx],
                yerr=[summary['pred_median'][sort_idx] - pred_16[sort_idx],
                      pred_84[sort_idx] - summary['pred_median'][sort_idx]],
                fmt='o', markersize=4, alpha=0.6, capsize=2, capthick=1,
                label='Predictions ± 68% CI')

    # Perfect prediction line
    age_range = [summary['true_age'].min(), summary['true_age'].max()]
    ax1.plot(age_range, age_range, 'r--', lw=2, label='Perfect prediction')

    # Add shaded region for ±0.1 dex
    ax1.fill_between(age_range,
                    [age_range[0]-0.1, age_range[1]-0.1],
                    [age_range[0]+0.1, age_range[1]+0.1],
                    alpha=0.2, color='red', label='±0.1 dex')

    ax1.set_xlabel('True log(Age) [dex]', fontsize=12)
    ax1.set_ylabel('Predicted log(Age) [dex]', fontsize=12)
    ax1.set_title('Predictions with 68% Confidence Intervals', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Model Uncertainty vs True Age ==========
    ax2 = axes[0, 1]

    # Create hexbin plot for density
    hexbin = ax2.hexbin(summary['true_age'], summary['model_uncertainty'],
                       gridsize=30, cmap='Blues', mincnt=1)

    # Add colorbar
    cb = plt.colorbar(hexbin, ax=ax2)
    cb.set_label('Number of stars', fontsize=10)

    # Add running median
    age_bins = np.linspace(summary['true_age'].min(), summary['true_age'].max(), 20)
    bin_centers = (age_bins[:-1] + age_bins[1:]) / 2
    binned_unc = []
    binned_unc_std = []

    for i in range(len(age_bins)-1):
        mask = (summary['true_age'] >= age_bins[i]) & (summary['true_age'] < age_bins[i+1])
        if mask.sum() > 3:
            binned_unc.append(np.median(summary['model_uncertainty'][mask]))
            binned_unc_std.append(np.std(summary['model_uncertainty'][mask]))
        else:
            binned_unc.append(np.nan)
            binned_unc_std.append(np.nan)

    ax2.plot(bin_centers, binned_unc, 'r-', lw=2, label='Median uncertainty')
    ax2.fill_between(bin_centers,
                    np.array(binned_unc) - np.array(binned_unc_std),
                    np.array(binned_unc) + np.array(binned_unc_std),
                    alpha=0.3, color='red', label='±1 std')

    ax2.set_xlabel('True log(Age) [dex]', fontsize=12)
    ax2.set_ylabel('Model Uncertainty [dex]', fontsize=12)
    ax2.set_title('Model Uncertainty vs True Age', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Total Uncertainty Breakdown ==========
    ax3 = axes[1, 0]

    # Sort by total uncertainty for stacked view
    sort_idx_unc = np.argsort(summary['total_predictive_uncertainty'])
    x_pos = np.arange(len(summary))

    # Create stacked bar chart
    ax3.bar(x_pos, summary['observational_uncertainty'][sort_idx_unc],
            label='Observational', alpha=0.8, color='skyblue')
    ax3.bar(x_pos, summary['model_uncertainty'][sort_idx_unc],
            bottom=summary['observational_uncertainty'][sort_idx_unc],
            label='Model', alpha=0.8, color='orange')
    ax3.bar(x_pos, summary['intrinsic_scatter'][sort_idx_unc],
            bottom=summary['observational_uncertainty'][sort_idx_unc] +
                   summary['model_uncertainty'][sort_idx_unc],
            label='Intrinsic', alpha=0.8, color='green')

    # Overlay total
    ax3.plot(x_pos, summary['total_predictive_uncertainty'][sort_idx_unc],
            'k-', lw=2, label='Total (quadrature sum)')

    ax3.set_xlabel('Star Index (sorted by total uncertainty)', fontsize=12)
    ax3.set_ylabel('Uncertainty [dex]', fontsize=12)
    ax3.set_title('Uncertainty Components Breakdown', fontsize=14)
    ax3.legend()
    ax3.set_xlim(0, len(summary))

    # ========== Plot 4: Residuals with Uncertainty Bands ==========
    ax4 = axes[1, 1]

    # Calculate normalized residuals
    residuals = summary['true_age'] - summary['pred_median']

    # Create scatter plot colored by uncertainty
    scatter = ax4.scatter(summary['true_age'], residuals,
                         c=summary['total_predictive_uncertainty'],
                         cmap='viridis', alpha=0.6, s=30)

    # Add colorbar
    cb2 = plt.colorbar(scatter, ax=ax4)
    cb2.set_label('Total Uncertainty [dex]', fontsize=10)

    # Add ±1σ and ±2σ bands based on median uncertainty
    median_unc = np.median(summary['total_predictive_uncertainty'])
    ax4.axhspan(-median_unc, median_unc, alpha=0.2, color='gray', label='±1σ (median)')
    ax4.axhspan(-2*median_unc, 2*median_unc, alpha=0.1, color='gray', label='±2σ (median)')
    ax4.axhline(y=0, color='red', linestyle='--', lw=2)

    ax4.set_xlabel('True log(Age) [dex]', fontsize=12)
    ax4.set_ylabel('Residual (True - Predicted) [dex]', fontsize=12)
    ax4.set_title('Residuals Colored by Total Uncertainty', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Bayesian Neural Network Age Predictions with Uncertainties', fontsize=16)
    plt.tight_layout()

    # Save figure
    plt.savefig('./BNN_targeted_output/predictions_with_uncertainties.png', dpi=150, bbox_inches='tight')
    print("Saved comprehensive uncertainty plot to ./BNN_targeted_output/predictions_with_uncertainties.png")

    # Print summary statistics
    print("\n" + "="*60)
    print("UNCERTAINTY STATISTICS:")
    print("="*60)
    print(f"Mean uncertainties:")
    print(f"  Observational: {summary['observational_uncertainty'].mean():.4f} ± {summary['observational_uncertainty'].std():.4f} dex")
    print(f"  Model:         {summary['model_uncertainty'].mean():.4f} ± {summary['model_uncertainty'].std():.4f} dex")
    print(f"  Intrinsic:     {summary['intrinsic_scatter'].mean():.4f} ± {summary['intrinsic_scatter'].std():.4f} dex")
    print(f"  Total:         {summary['total_predictive_uncertainty'].mean():.4f} ± {summary['total_predictive_uncertainty'].std():.4f} dex")

    # Check if uncertainties vary with age
    corr_unc_age = np.corrcoef(summary['true_age'], summary['model_uncertainty'])[0,1]
    print(f"\nCorrelation between age and model uncertainty: {corr_unc_age:.3f}")

    # High age stars
    high_age_mask = summary['true_age'] > 1.0
    if high_age_mask.sum() > 0:
        print(f"\nHigh age stars (logAge > 1.0):")
        print(f"  Model uncertainty: {summary[high_age_mask]['model_uncertainty'].mean():.4f} ± {summary[high_age_mask]['model_uncertainty'].std():.4f} dex")
        print(f"  Total uncertainty: {summary[high_age_mask]['total_predictive_uncertainty'].mean():.4f} ± {summary[high_age_mask]['total_predictive_uncertainty'].std():.4f} dex")

    plt.show()

    return summary

if __name__ == "__main__":
    summary = plot_predictions_with_uncertainties()
