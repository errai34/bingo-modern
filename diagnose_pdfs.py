"""
Diagnostic script to visualize posterior PDFs for selected stars
Shows the full posterior distribution for each star along with true age
"""

import numpy as np
import pandas as pd
import torch
import pyro
import matplotlib.pyplot as plt
from scipy import stats
from train_bnn import (
    BayesianNeuralNetwork,
    load_astronomical_data,
    get_targeted_posterior_samples
)
from pyro.infer.autoguide import AutoDiagonalNormal
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_star_pdfs(model, guide, X_test, X_err_test, y_test, y_err_test,
                   star_indices=None, num_samples=10000):
    """Plot posterior PDFs for selected stars"""

    if star_indices is None:
        # Select diverse stars: young, middle-aged, old, and high uncertainty
        ages = y_test
        uncertainties = y_err_test

        # Find indices for different types
        young_idx = np.argmin(np.abs(ages - 0.0))  # Young star
        middle_idx = np.argmin(np.abs(ages - 0.7))  # Middle-aged
        old_idx = np.argmin(np.abs(ages - 1.0))  # Old star
        very_old_idx = np.argmax(ages)  # Oldest star
        high_unc_idx = np.argmax(uncertainties)  # Highest uncertainty

        star_indices = [young_idx, middle_idx, old_idx, very_old_idx, high_unc_idx]
        labels = ['Young Star', 'Middle-Aged Star', 'Old Star', 'Oldest Star', 'High Uncertainty Star']
    else:
        labels = [f'Star {i}' for i in star_indices]

    # Get posterior samples for these stars
    X_subset = torch.FloatTensor(X_test[star_indices]).to(device)
    X_err_subset = torch.FloatTensor(X_err_test[star_indices]).to(device)
    y_err_subset = torch.FloatTensor(y_err_test[star_indices]).to(device)

    print(f"Generating {num_samples} posterior samples for {len(star_indices)} stars...")

    # Get full posterior samples
    total_samples, predictions, model_uncertainty, intrinsic_scatter = get_targeted_posterior_samples(
        model, guide, X_subset, X_err_subset, y_err_subset, num_samples=num_samples
    )

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (idx, label) in enumerate(zip(star_indices, labels)):
        if i >= 6:  # Only plot first 6
            break

        ax = axes[i]

        # Get samples for this star
        star_samples = total_samples[:, i]
        mean_pred = predictions[:, i]
        model_unc = model_uncertainty[:, i]

        # True values
        true_age = y_test[idx]
        true_err = y_err_test[idx]

        # Plot histogram of full posterior
        counts, bins, _ = ax.hist(star_samples, bins=50, alpha=0.7, density=True,
                                 label='Full Posterior', color='skyblue', edgecolor='navy')

        # Overlay KDE for smoother visualization
        kde = stats.gaussian_kde(star_samples)
        x_range = np.linspace(star_samples.min() - 0.2, star_samples.max() + 0.2, 1000)
        ax.plot(x_range, kde(x_range), 'b-', lw=2, label='Posterior KDE')

        # Plot true value
        ax.axvline(true_age, color='red', linestyle='--', linewidth=2, label=f'True: {true_age:.3f}')

        # Plot observational uncertainty range
        ax.axvspan(true_age - true_err, true_age + true_err, alpha=0.2, color='red',
                  label=f'Obs. Unc.: ±{true_err:.3f}')

        # Add prediction statistics
        pred_median = np.median(star_samples)
        pred_mean = np.mean(star_samples)
        pred_std = np.std(star_samples)
        lower_ci = np.percentile(star_samples, 16)
        upper_ci = np.percentile(star_samples, 84)

        # Mean prediction from model only (without full uncertainty)
        mean_only = np.median(mean_pred)

        # Add vertical lines for key statistics
        ax.axvline(pred_median, color='green', linestyle='-', linewidth=1.5,
                  label=f'Median: {pred_median:.3f}')
        ax.axvline(mean_only, color='orange', linestyle=':', linewidth=1.5,
                  label=f'Mean pred: {mean_only:.3f}')

        # Shade 68% CI
        ax.axvspan(lower_ci, upper_ci, alpha=0.1, color='green')

        # Labels and title
        ax.set_xlabel('log(Age) [dex]')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{label}\nTrue: {true_age:.3f}, Pred: {pred_median:.3f}±{pred_std:.3f}')
        ax.legend(loc='upper right', fontsize=8)

        # Add text box with statistics
        stats_text = f'Model Unc: {np.mean(model_unc):.3f}\n'
        stats_text += f'Intrinsic: {np.mean(intrinsic_scatter):.3f}\n'
        stats_text += f'Total Unc: {pred_std:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove empty subplot if we have 5 stars
    if len(star_indices) < 6:
        fig.delaxes(axes[5])

    plt.suptitle('Posterior Age PDFs for Selected Stars', fontsize=16)
    plt.tight_layout()

    return fig, star_indices

def main():
    """Main diagnostic function"""

    print("="*60)
    print("DIAGNOSTIC: Posterior PDFs for Individual Stars")
    print("="*60)

    # Paths
    test_path = './test_data/TestOriginalNorm_dr17.csv'
    param_path = './BNN_targeted_output/targeted_bnn_params.pth'

    # Load test data
    X_test, X_err_test, y_test, y_err_test = load_astronomical_data(test_path)
    print(f"\nLoaded {len(X_test)} test stars")
    print(f"Age range: [{y_test.min():.3f}, {y_test.max():.3f}]")

    # Initialize model (same architecture as training)
    model = BayesianNeuralNetwork(
        input_dim=X_test.shape[1],
        hidden_dim=16,
        use_skip_connections=True,
        use_empirical_output_bias=True,
        use_leaky_relu=True,
        y_mean=0.691,  # From training data
        y_std=0.339     # From training data
    )
    model.to(device)

    # Load trained parameters
    print(f"\nLoading model parameters from {param_path}")
    pyro.get_param_store().load(param_path)

    # Create guide
    guide = AutoDiagonalNormal(model)

    # Plot PDFs for selected diverse stars
    fig1, selected_indices = plot_star_pdfs(
        model, guide, X_test, X_err_test, y_test, y_err_test,
        star_indices=None,  # Auto-select diverse stars
        num_samples=10000
    )

    plt.savefig('./BNN_targeted_output/diagnostic_pdfs_diverse.png', dpi=150, bbox_inches='tight')
    print("\nSaved diverse star PDFs to ./BNN_targeted_output/diagnostic_pdfs_diverse.png")

    # Also plot some random stars
    np.random.seed(42)
    random_indices = np.random.choice(len(X_test), size=5, replace=False)

    fig2, _ = plot_star_pdfs(
        model, guide, X_test, X_err_test, y_test, y_err_test,
        star_indices=random_indices,
        num_samples=10000
    )

    plt.savefig('./BNN_targeted_output/diagnostic_pdfs_random.png', dpi=150, bbox_inches='tight')
    print("Saved random star PDFs to ./BNN_targeted_output/diagnostic_pdfs_random.png")

    # Print summary statistics for selected stars
    print("\n" + "="*60)
    print("SUMMARY OF SELECTED STARS:")
    print("="*60)

    for i, idx in enumerate(selected_indices):
        print(f"\nStar {i+1} (Index {idx}):")
        print(f"  True Age: {y_test[idx]:.3f} ± {y_err_test[idx]:.3f}")
        print(f"  Features: {X_test[idx]}")

    plt.show()

if __name__ == "__main__":
    main()
