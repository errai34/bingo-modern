import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# USER CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# =============================================================================
INPUT_FOLDER = 'BNN_targeted_output'  # Change this to your desired input folder
OUTPUT_FOLDER = '.'                   # Change this to your desired output folder
CSV_FILENAME = 'targeted_prediction_summary.csv'
# =============================================================================

# Load the results
csv_path = f'{INPUT_FOLDER}/{CSV_FILENAME}'
df = pd.read_csv(csv_path)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All stars
ax1.scatter(df['true_age'], df['pred_median'], alpha=0.6, s=20, c='blue', label='All stars')
# Add diagonal line
age_min, age_max = df['true_age'].min(), df['true_age'].max()
ax1.plot([age_min, age_max], [age_min, age_max], 'r--', lw=2, label='Perfect prediction')
# Add high-age threshold
ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='High age threshold')
ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)

ax1.set_xlabel('True logAge', fontsize=12)
ax1.set_ylabel('Predicted logAge', fontsize=12)
ax1.set_title('True vs Predicted Ages - All Stars', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Calculate statistics for text
mae = np.abs(df['true_age'] - df['pred_median']).mean()
rms = np.sqrt(((df['true_age'] - df['pred_median'])**2).mean())
corr = np.corrcoef(df['true_age'], df['pred_median'])[0,1]

ax1.text(0.05, 0.95, f'MAE: {mae:.4f} dex\nRMS: {rms:.4f} dex\nCorr: {corr:.3f}',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Color-coded by age with residuals
scatter = ax2.scatter(df['true_age'], df['pred_median'],
                     c=df['true_age'], cmap='viridis',
                     alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
ax2.plot([age_min, age_max], [age_min, age_max], 'r--', lw=2, label='Perfect prediction')

# Highlight high-age stars
high_age_mask = df['true_age'] > 1
ax2.scatter(df[high_age_mask]['true_age'], df[high_age_mask]['pred_median'],
           s=100, facecolors='none', edgecolors='red', linewidth=2,
           label=f'High age stars (n={high_age_mask.sum()})')

ax2.set_xlabel('True logAge', fontsize=12)
ax2.set_ylabel('Predicted logAge', fontsize=12)
ax2.set_title('True vs Predicted Ages - Color by Age', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('True logAge', fontsize=10)

# Calculate high-age statistics
if high_age_mask.sum() > 0:
    high_age_df = df[high_age_mask]
    high_mae = np.abs(high_age_df['true_age'] - high_age_df['pred_median']).mean()
    high_bias = (high_age_df['true_age'] - high_age_df['pred_median']).mean()

    ax2.text(0.05, 0.95, f'High age (>1) stats:\nMAE: {high_mae:.4f} dex\nBias: {high_bias:.4f} dex',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))

plt.tight_layout()
output_path = f'{OUTPUT_FOLDER}/age_predictions_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# Additional residual analysis
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

# Residuals vs true age
residuals = df['true_age'] - df['pred_median']
ax3.scatter(df['true_age'], residuals, alpha=0.6, s=20)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)

# Add running mean
from scipy.ndimage import uniform_filter1d
sorted_idx = np.argsort(df['true_age'])
sorted_age = df['true_age'].values[sorted_idx]
sorted_residuals = residuals.values[sorted_idx]
window = 50
if len(sorted_age) > window:
    running_mean = uniform_filter1d(sorted_residuals, size=window, mode='nearest')
    ax3.plot(sorted_age, running_mean, 'g-', lw=3, label='Running mean')

ax3.set_xlabel('True logAge', fontsize=12)
ax3.set_ylabel('Residual (True - Predicted)', fontsize=12)
ax3.set_title('Residuals vs True Age', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Histogram of residuals by age group
ax4.hist(residuals[df['true_age'] <= 1], bins=30, alpha=0.6, label='Low age (≤1)', density=True)
if high_age_mask.sum() > 0:
    ax4.hist(residuals[high_age_mask], bins=20, alpha=0.6, label='High age (>1)', density=True)

ax4.axvline(x=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Residual (True - Predicted)', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('Distribution of Residuals', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = f'{OUTPUT_FOLDER}/residuals_analysis.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved to:")
print(f"  - {output_path}")
print(f"  - {output_path2}")
