import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

# =============================================================================
# USER CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# =============================================================================
INPUT_FOLDER = 'BNN_targeted_output'  # Change this to your desired input folder
OUTPUT_FOLDER = '.'                   # Change this to your desired output folder
CSV_FILENAME = 'targeted_prediction_summary.csv'
# =============================================================================

def plot_predicted_vs_true_comprehensive():
    """Create comprehensive predicted vs true age plots"""

    print("="*60)
    print("PREDICTED vs TRUE AGE ANALYSIS")
    print("="*60)

    # Load retrained results
    csv_path = f'{INPUT_FOLDER}/{CSV_FILENAME}'
    summary = pd.read_csv(csv_path)

    true_ages = summary['true_age'].values
    pred_ages = summary['pred_median'].values
    uncertainties = summary['total_predictive_uncertainty'].values
    obs_unc = summary['observational_uncertainty'].values
    model_unc = summary['model_uncertainty'].values
    residuals = summary['residual'].values
    norm_residuals = summary['normalized_residual'].values

    # Calculate statistics
    correlation = np.corrcoef(true_ages, pred_ages)[0, 1]
    mae = np.abs(residuals).mean()
    rms = np.sqrt((residuals**2).mean())
    within_1sigma = (np.abs(norm_residuals) < 1).mean()
    within_2sigma = (np.abs(norm_residuals) < 2).mean()

    print(f"Dataset: {len(true_ages)} stars")
    print(f"Correlation: {correlation:.3f}")
    print(f"MAE: {mae:.3f} dex")
    print(f"RMS: {rms:.3f} dex")
    print(f"Within 1σ: {within_1sigma:.1%}")
    print(f"Within 2σ: {within_2sigma:.1%}")

    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 15))

    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main scatter plot with error bars
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot with error bars
    ax1.errorbar(true_ages, pred_ages, yerr=uncertainties,
                fmt='o', alpha=0.6, markersize=4, elinewidth=1, capsize=2,
                color='steelblue', ecolor='lightblue', label='Predictions ± σ')

    # Perfect prediction line
    age_range = [min(true_ages.min(), pred_ages.min()),
                 max(true_ages.max(), pred_ages.max())]
    ax1.plot(age_range, age_range, 'r--', lw=2, label='Perfect prediction')

    # Add statistics text
    stats_text = f'Corr: {correlation:.3f}\nMAE: {mae:.3f}\nRMS: {rms:.3f}\n1σ: {within_1sigma:.1%}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10)

    ax1.set_xlabel('True log(Age)')
    ax1.set_ylabel('Predicted log(Age)')
    ax1.set_title('Predicted vs True Age (with uncertainties)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. Color-coded by uncertainty
    ax2 = fig.add_subplot(gs[0, 1])

    scatter = ax2.scatter(true_ages, pred_ages, c=uncertainties,
                         s=30, alpha=0.7, cmap='viridis')
    ax2.plot(age_range, age_range, 'r--', lw=2, label='Perfect prediction')

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Total Uncertainty (dex)')

    ax2.set_xlabel('True log(Age)')
    ax2.set_ylabel('Predicted log(Age)')
    ax2.set_title('Color-coded by Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # 3. Color-coded by age groups
    ax3 = fig.add_subplot(gs[0, 2])

    # Define age groups
    young_mask = true_ages < 0.0
    middle_mask = (true_ages >= 0.0) & (true_ages <= 1.0)
    old_mask = true_ages > 1.0

    ax3.scatter(true_ages[young_mask], pred_ages[young_mask],
               c='blue', s=30, alpha=0.7, label=f'Young (<1 Gyr, n={young_mask.sum()})')
    ax3.scatter(true_ages[middle_mask], pred_ages[middle_mask],
               c='green', s=30, alpha=0.7, label=f'Middle (1-10 Gyr, n={middle_mask.sum()})')
    ax3.scatter(true_ages[old_mask], pred_ages[old_mask],
               c='red', s=30, alpha=0.7, label=f'Old (>10 Gyr, n={old_mask.sum()})')

    ax3.plot(age_range, age_range, 'k--', lw=2, label='Perfect prediction')

    ax3.set_xlabel('True log(Age)')
    ax3.set_ylabel('Predicted log(Age)')
    ax3.set_title('Color-coded by Age Groups')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # 4. Residuals vs True Age
    ax4 = fig.add_subplot(gs[1, 0])

    ax4.scatter(true_ages, residuals, alpha=0.6, s=20, color='purple')
    ax4.axhline(0, color='red', linestyle='--', lw=2)
    ax4.axhline(mae, color='orange', linestyle=':', lw=1, label=f'±MAE ({mae:.3f})')
    ax4.axhline(-mae, color='orange', linestyle=':', lw=1)

    ax4.set_xlabel('True log(Age)')
    ax4.set_ylabel('Residual (True - Pred)')
    ax4.set_title('Residuals vs True Age')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Residuals vs Predicted Age
    ax5 = fig.add_subplot(gs[1, 1])

    ax5.scatter(pred_ages, residuals, alpha=0.6, s=20, color='darkgreen')
    ax5.axhline(0, color='red', linestyle='--', lw=2)
    ax5.axhline(mae, color='orange', linestyle=':', lw=1, label=f'±MAE ({mae:.3f})')
    ax5.axhline(-mae, color='orange', linestyle=':', lw=1)

    ax5.set_xlabel('Predicted log(Age)')
    ax5.set_ylabel('Residual (True - Pred)')
    ax5.set_title('Residuals vs Predicted Age')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Normalized residuals histogram
    ax6 = fig.add_subplot(gs[1, 2])

    ax6.hist(norm_residuals, bins=30, density=True, alpha=0.7,
            color='lightcoral', edgecolor='darkred', label='Normalized Residuals')

    # Plot ideal normal distribution
    x_norm = np.linspace(-4, 4, 100)
    y_norm = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x_norm**2)
    ax6.plot(x_norm, y_norm, 'k-', lw=2, label='N(0,1)')

    ax6.axvline(0, color='red', linestyle='--', lw=1)
    ax6.axvline(1, color='orange', linestyle=':', lw=1, label='±1σ')
    ax6.axvline(-1, color='orange', linestyle=':', lw=1)

    ax6.set_xlabel('Normalized Residuals')
    ax6.set_ylabel('Density')
    ax6.set_title(f'Calibration Check ({within_1sigma:.1%} within 1σ)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Uncertainty vs Age
    ax7 = fig.add_subplot(gs[2, 0])

    ax7.scatter(true_ages, obs_unc, alpha=0.6, s=15, color='blue', label='Observational')
    ax7.scatter(true_ages, model_unc, alpha=0.6, s=15, color='orange', label='Model')
    ax7.scatter(true_ages, uncertainties, alpha=0.6, s=15, color='red', label='Total')

    ax7.set_xlabel('True log(Age)')
    ax7.set_ylabel('Uncertainty (dex)')
    ax7.set_title('Uncertainties vs Age')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Q-Q plot for normality check
    ax8 = fig.add_subplot(gs[2, 1])

    stats.probplot(norm_residuals, dist="norm", plot=ax8)
    ax8.set_title('Q-Q Plot (Normality Check)')
    ax8.grid(True, alpha=0.3)

    # 9. Performance by age bins
    ax9 = fig.add_subplot(gs[2, 2])

    # Bin by age
    age_bins = pd.cut(true_ages, bins=5, labels=['Very Young', 'Young', 'Middle', 'Old', 'Very Old'])
    bin_stats = []
    bin_names = []

    for bin_name in ['Very Young', 'Young', 'Middle', 'Old', 'Very Old']:
        mask = age_bins == bin_name
        if mask.any():
            bin_mae = np.abs(residuals[mask]).mean()
            bin_within_1sigma = (np.abs(norm_residuals[mask]) < 1).mean()
            bin_stats.append([bin_mae, bin_within_1sigma])
            bin_names.append(bin_name)

    bin_stats = np.array(bin_stats)

    x_pos = np.arange(len(bin_names))
    width = 0.35

    ax9_twin = ax9.twinx()

    bars1 = ax9.bar(x_pos - width/2, bin_stats[:, 0], width,
                   label='MAE (dex)', color='skyblue', alpha=0.7)
    bars2 = ax9_twin.bar(x_pos + width/2, bin_stats[:, 1], width,
                        label='Frac within 1σ', color='lightcoral', alpha=0.7)

    ax9.set_xlabel('Age Groups')
    ax9.set_ylabel('MAE (dex)', color='blue')
    ax9_twin.set_ylabel('Fraction within 1σ', color='red')
    ax9.set_title('Performance by Age Groups')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(bin_names, rotation=45)
    ax9.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax9.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.002,
                f'{bin_stats[i, 0]:.3f}', ha='center', va='bottom', fontsize=8)
        ax9_twin.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                     f'{bin_stats[i, 1]:.1%}', ha='center', va='bottom', fontsize=8)

    ax9.legend(loc='upper left')
    ax9_twin.legend(loc='upper right')

    plt.suptitle(f'Retrained BNN: Predicted vs True Age Analysis (N={len(true_ages)} stars)',
                fontsize=16, y=0.98)

    output_path = f'{OUTPUT_FOLDER}/predicted_vs_true_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_simple_predicted_vs_true():
    """Create a clean, publication-ready predicted vs true plot"""

    print("\nCreating publication-ready plot...")

    # Load data
    csv_path = f'{INPUT_FOLDER}/{CSV_FILENAME}'
    summary = pd.read_csv(csv_path)

    true_ages = summary['true_age'].values
    pred_ages = summary['pred_median'].values
    uncertainties = summary['total_predictive_uncertainty'].values
    residuals = summary['residual'].values

    # Calculate statistics
    correlation = np.corrcoef(true_ages, pred_ages)[0, 1]
    mae = np.abs(residuals).mean()
    rms = np.sqrt((residuals**2).mean())
    within_1sigma = (np.abs(summary['normalized_residual']) < 1).mean()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Main plot with error bars
    ax1.errorbar(true_ages, pred_ages, yerr=uncertainties,
                fmt='o', alpha=0.6, markersize=4, elinewidth=0.8, capsize=1.5,
                color='steelblue', ecolor='lightsteelblue', label='BNN Predictions')

    # Perfect prediction line
    age_range = [min(true_ages.min(), pred_ages.min()) - 0.1,
                 max(true_ages.max(), pred_ages.max()) + 0.1]
    ax1.plot(age_range, age_range, 'r--', lw=2, label='Perfect Prediction', alpha=0.8)

    # Add ±1σ region around perfect line
    y_upper = np.array(age_range) + mae
    y_lower = np.array(age_range) - mae
    ax1.fill_between(age_range, y_lower, y_upper, alpha=0.2, color='red',
                    label=f'±MAE ({mae:.3f} dex)')

    # Statistics box
    stats_text = f'N = {len(true_ages)} stars\n'
    stats_text += f'r = {correlation:.3f}\n'
    stats_text += f'MAE = {mae:.3f} dex\n'
    stats_text += f'RMS = {rms:.3f} dex\n'
    stats_text += f'{within_1sigma:.1%} within 1σ'

    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
            verticalalignment='top', fontsize=11, fontweight='bold')

    ax1.set_xlabel('True log(Age)', fontsize=12)
    ax1.set_ylabel('Predicted log(Age)', fontsize=12)
    ax1.set_title('Retrained BNN: Predicted vs True Age', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Set equal axis limits
    all_ages = np.concatenate([true_ages, pred_ages])
    ax1.set_xlim(all_ages.min() - 0.1, all_ages.max() + 0.1)
    ax1.set_ylim(all_ages.min() - 0.1, all_ages.max() + 0.1)

    # Residual plot
    ax2.scatter(true_ages, residuals, alpha=0.6, s=25, color='darkblue', edgecolors='white', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='-', lw=2, alpha=0.8, label='Zero residual')
    ax2.axhline(mae, color='orange', linestyle='--', lw=1.5, alpha=0.8, label=f'±MAE ({mae:.3f})')
    ax2.axhline(-mae, color='orange', linestyle='--', lw=1.5, alpha=0.8)

    # Add ±1σ and ±2σ regions
    ax2.axhspan(-mae, mae, alpha=0.1, color='orange', label='±MAE region')
    ax2.axhspan(-2*mae, 2*mae, alpha=0.05, color='red', label='±2×MAE region')

    ax2.set_xlabel('True log(Age)', fontsize=12)
    ax2.set_ylabel('Residual (True - Predicted)', fontsize=12)
    ax2.set_title('Residuals vs True Age', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'{OUTPUT_FOLDER}/predicted_vs_true_clean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_predicted_vs_true_comprehensive()
    plot_simple_predicted_vs_true()
