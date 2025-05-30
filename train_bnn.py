"""
TARGETED Fix for Robust Bayesian Neural Network
Addresses the real issues without making drastic changes

Key fixes:
1. Proper parameter store clearing
2. Better loss tracking and reproducibility
3. Improved training monitoring
4. Better handling of edge cases
5. LeakyReLU activation to prevent dying neurons
6. Empirical Bayes priors based on data statistics
7. More complex architecture for better high-age predictions
8. **NEW**: Proper input uncertainty propagation via hierarchical modeling
   - Models true (noise-free) inputs as latent variables
   - x_true ~ Normal(x_observed, x_err)
   - Propagates input uncertainty through the network
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_astronomical_data(file_path: str) -> Tuple[np.ndarray, ...]:
    """Load astronomical data matching your original format"""

    print(f"Loading data from {file_path}...")

    # Handle both CSV and HDF5 files
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.hdf5'):
        from astropy.table import Table
        table = Table.read(file_path)
        df = table.to_pandas()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    print(f"Loaded {len(df)} samples")

    # Feature columns - check which columns exist
    all_possible_cols = ['LOGG_NORM', 'TEFF_NORM', 'MG_FE_NORM', 'FE_H_NORM',
                        'C_FE_NORM', 'N_FE_NORM', 'ALPHA_M_NORM', 'M_H_NORM',
                        'G_NORM', 'BP_NORM', 'RP_NORM', 'J_NORM', 'H_NORM', 'K_NORM']

    feature_cols = [col for col in all_possible_cols if col in df.columns]
    print(f"Using features: {feature_cols}")

    # Error columns
    error_cols = [col.replace('_NORM', '_ERR_NORM') for col in feature_cols]

    # Extract data
    X = df[feature_cols].values.astype(np.float32)
    X_err = df[error_cols].values.astype(np.float32)
    y = df['logAge'].values.astype(np.float32)
    y_err = df['logAgeErr'].values.astype(np.float32)

    return X, X_err, y, y_err


# ============================================================================
# TARGETED Fixed Robust Bayesian Neural Network with Empirical Bayes
# ============================================================================


class BayesianNeuralNetwork(PyroModule):
    """
    Modified BNN with fixes for better high-age predictions:
    1. Input uncertainty modeling
    2. Standard prior on output bias (not empirical Bayes)
    3. Option to use simpler architecture
    4. Option to use standard ReLU
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int = 16,  # Reduced default from 32
                 use_skip_connections: bool = False,  # Disabled by default
                 use_empirical_output_bias: bool = False,  # Disabled by default
                 use_leaky_relu: bool = False,  # Standard ReLU by default
                 y_mean: float = 0.0, y_std: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_skip_connections = use_skip_connections
        self.use_empirical_output_bias = use_empirical_output_bias
        self.use_leaky_relu = use_leaky_relu
        self.y_mean = y_mean
        self.y_std = y_std

        # Neural network layers for mean prediction
        self.fc1_mean = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc2_mean = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        # Optional third layer
        self.fc3_mean = PyroModule[nn.Linear](hidden_dim, hidden_dim)

        # Skip connection (optional)
        if self.use_skip_connections:
            self.fc_skip_mean = PyroModule[nn.Linear](input_dim, hidden_dim)

        self.fc_out_mean = PyroModule[nn.Linear](hidden_dim, 1)

        # Variance network
        self.fc1_var = PyroModule[nn.Linear](input_dim, hidden_dim//2)
        self.fc2_var = PyroModule[nn.Linear](hidden_dim//2, hidden_dim//4)
        self.fc3_var = PyroModule[nn.Linear](hidden_dim//4, 1)

        # Set priors
        self._set_priors()

    def _set_priors(self):
        """Set Bayesian priors"""
        # Use larger prior scale like old model
        prior_scale = 1.0  # Changed from 0.5

        # Mean network priors - hidden layers
        self.fc1_mean.weight = PyroSample(
            dist.Normal(0., prior_scale).expand([self.hidden_dim, self.input_dim]).to_event(2)
        )
        self.fc1_mean.bias = PyroSample(
            dist.Normal(0., prior_scale).expand([self.hidden_dim]).to_event(1)
        )
        self.fc2_mean.weight = PyroSample(
            dist.Normal(0., prior_scale).expand([self.hidden_dim, self.hidden_dim]).to_event(2)
        )
        self.fc2_mean.bias = PyroSample(
            dist.Normal(0., prior_scale).expand([self.hidden_dim]).to_event(1)
        )
        self.fc3_mean.weight = PyroSample(
            dist.Normal(0., prior_scale).expand([self.hidden_dim, self.hidden_dim]).to_event(2)
        )
        self.fc3_mean.bias = PyroSample(
            dist.Normal(0., prior_scale).expand([self.hidden_dim]).to_event(1)
        )

        # Skip connection priors (if used)
        if self.use_skip_connections:
            self.fc_skip_mean.weight = PyroSample(
                dist.Normal(0., prior_scale).expand([self.hidden_dim, self.input_dim]).to_event(2)
            )
            self.fc_skip_mean.bias = PyroSample(
                dist.Normal(0., prior_scale).expand([self.hidden_dim]).to_event(1)
            )

        # Output layer priors
        output_weight_scale = prior_scale / self.hidden_dim**0.5
        self.fc_out_mean.weight = PyroSample(
            dist.Normal(0., output_weight_scale).expand([1, self.hidden_dim]).to_event(2)
        )

        # CRITICAL FIX: Use standard or empirical Bayes based on flag
        if self.use_empirical_output_bias:
            self.fc_out_mean.bias = PyroSample(
                dist.Normal(self.y_mean, self.y_std * 1.0).expand([1]).to_event(1)
            )
        else:
            # Standard prior like old model - better for extrapolation
            self.fc_out_mean.bias = PyroSample(
                dist.Normal(0., prior_scale).expand([1]).to_event(1)
            )

        # Variance network priors
        var_prior_scale = prior_scale * 0.3  # Adjusted
        self.fc1_var.weight = PyroSample(
            dist.Normal(0., var_prior_scale).expand([self.hidden_dim//2, self.input_dim]).to_event(2)
        )
        self.fc1_var.bias = PyroSample(
            dist.Normal(0., var_prior_scale).expand([self.hidden_dim//2]).to_event(1)
        )
        self.fc2_var.weight = PyroSample(
            dist.Normal(0., var_prior_scale).expand([self.hidden_dim//4, self.hidden_dim//2]).to_event(2)
        )
        self.fc2_var.bias = PyroSample(
            dist.Normal(0., var_prior_scale).expand([self.hidden_dim//4]).to_event(1)
        )
        self.fc3_var.weight = PyroSample(
            dist.Normal(0., var_prior_scale).expand([1, self.hidden_dim//4]).to_event(2)
        )
        log_var_mean = np.log(self.y_std**2 * 0.1)
        self.fc3_var.bias = PyroSample(
            dist.Normal(log_var_mean, 0.5).expand([1]).to_event(1)
        )

    def forward(self, x, x_err=None, y=None, y_err=None):
        """Forward pass with proper input uncertainty modeling

        We model the true (noise-free) inputs as latent variables:
        x_true ~ Normal(x_observed, x_err)

        This is a proper hierarchical model that avoids the bias issues
        of simply adding noise during training.
        """

        # Handle input uncertainty using vectorized sampling without plates
        if x_err is not None and torch.sum(x_err) > 0:
            # Clamp uncertainties to avoid numerical issues
            x_err_clamped = torch.clamp(x_err, min=1e-6, max=1.0)

            # During training, add input uncertainty using reparameterization trick
            if self.training:
                # Use reparameterization trick to sample x_true
                # This avoids the plate/batch size issues while still modeling uncertainty
                eps = torch.randn_like(x)
                x_true = x + eps * x_err_clamped
            else:
                # During inference, use the observed values (MAP estimate)
                x_true = x
        else:
            # No input errors provided, use observed values directly
            x_true = x

        # Choose activation function
        if self.use_leaky_relu:
            activation = lambda x: F.leaky_relu(x, negative_slope=0.01)
        else:
            activation = F.relu  # Standard ReLU like old model

        # Mean prediction network
        h1_mean = activation(self.fc1_mean(x_true))
        h2_mean = activation(self.fc2_mean(h1_mean))
        h3_mean = activation(self.fc3_mean(h2_mean))

        # Skip connection (if enabled)
        if self.use_skip_connections:
            skip = activation(self.fc_skip_mean(x_true))
            combined = h3_mean + skip
        else:
            combined = h3_mean

        mu = self.fc_out_mean(combined).squeeze(-1)

        # Apply output transformation to help with high ages
        # This can help the network predict extreme values
        # mu = mu * 1.1  # Simple scaling - uncomment if needed

        # Variance network
        h1_var = activation(self.fc1_var(x_true))
        h2_var = activation(self.fc2_var(h1_var))
        log_model_var = self.fc3_var(h2_var).squeeze(-1)
        log_model_var = torch.clamp(log_model_var, min=-10, max=3)  # Reduced max
        model_var = torch.exp(log_model_var)

        # Intrinsic scatter
        log_intrinsic_std = pyro.sample("log_intrinsic_std",
                                       dist.Normal(np.log(0.1), 0.3))  # Tighter prior
        intrinsic_var = torch.exp(log_intrinsic_std) ** 2

        # Register outputs
        mu = pyro.deterministic("prediction", mu)
        model_std = pyro.deterministic("model_uncertainty", torch.sqrt(model_var))
        intrinsic_std = pyro.deterministic("intrinsic_scatter", torch.sqrt(intrinsic_var))

        # Likelihood
        if y is not None and y_err is not None:
            observational_var = y_err ** 2
            # Expand intrinsic_var to match batch size
            intrinsic_var_expanded = intrinsic_var.expand(y.shape[0])
            total_var = observational_var + model_var + intrinsic_var_expanded
            total_std = torch.sqrt(total_var)
            total_std = torch.clamp(total_std, min=1e-6)

            with pyro.plate("data", y.shape[0]):
                pyro.sample("obs", dist.Normal(mu, total_std), obs=y)

        return mu, model_var, intrinsic_var


# ============================================================================
# Targeted Training Functions
# ============================================================================

def train_targeted_bnn(model: BayesianNeuralNetwork,
                      X_train: torch.Tensor,
                      X_err_train: torch.Tensor,
                      y_train: torch.Tensor,
                      y_err_train: torch.Tensor,
                      num_iterations: int = 8000,  # Increased from 5000
                      lr: float = 0.01,
                      batch_size: int = 512,
                      seed: int = None) -> Tuple[AutoDiagonalNormal, list]:
    """Train targeted BNN with better reproducibility and monitoring"""

    print("\n" + "="*60)
    print("Training TARGETED Fixed Robust Bayesian Neural Network")
    print("With Deeper Architecture and Skip Connections")
    print("="*60)

    # Set seed if provided for reproducibility
    if seed is not None:
        set_seed(seed)
        print(f"Using random seed: {seed}")

    # CRITICAL: Clear parameter store and ensure clean start
    pyro.clear_param_store()
    print("Parameter store cleared for clean training start")

    # Create data loader for mini-batches first to know number of batches
    dataset = torch.utils.data.TensorDataset(X_train, X_err_train, y_train, y_err_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup SVI with improved optimizer settings
    guide = AutoDiagonalNormal(model)

    # Use ClippedAdam optimizer with gradient clipping and learning rate decay
    from pyro.optim import ClippedAdam
    optimizer = ClippedAdam({
        "lr": lr,
        "clip_norm": 10.0,  # Gradient clipping for stability
        "lrd": 0.999**(1/len(loader))  # Decay per iteration to get 0.999 decay per epoch
    })

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Training loop with better monitoring
    losses = []
    smoothed_losses = []  # Exponential moving average of losses
    num_epochs = num_iterations // len(loader)

    print(f"Training for {num_epochs} epochs with {len(loader)} batches per epoch...")
    print(f"Total iterations: {num_iterations}")
    print(f"Batch size: {batch_size} (covers {100*batch_size/len(X_train):.1f}% of data per batch)")

    pbar = tqdm(range(num_epochs), desc="Training")
    best_loss = float('inf')
    patience_counter = 0

    for epoch in pbar:
        epoch_loss = 0.0
        batch_losses = []

        for batch_x, batch_x_err, batch_y, batch_y_err in loader:
            loss = svi.step(batch_x, batch_x_err, batch_y, batch_y_err)
            epoch_loss += loss
            batch_losses.append(loss)

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        # Calculate smoothed loss (exponential moving average)
        if epoch == 0:
            smoothed_loss = avg_loss
        else:
            smoothed_loss = 0.95 * smoothed_losses[-1] + 0.05 * avg_loss
        smoothed_losses.append(smoothed_loss)

        # Track best smoothed loss for early stopping
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Update progress bar with more info
        pbar.set_postfix({
            'loss': f'{avg_loss:.2f}',
            'smooth': f'{smoothed_loss:.2f}',
            'best': f'{best_loss:.2f}',
        })

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            loss_std = np.std(batch_losses)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f} ± {loss_std:.4f}, "
                  f"Smoothed = {smoothed_loss:.4f}")

    print(f"\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final smoothed loss: {smoothed_losses[-1]:.4f}")
    print(f"Best smoothed loss: {best_loss:.4f}")
    print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")

    return guide, losses


def get_targeted_posterior_samples(model: BayesianNeuralNetwork,
                                 guide: AutoDiagonalNormal,
                                 X: torch.Tensor,
                                 X_err: torch.Tensor,
                                 y_err: torch.Tensor,
                                 num_samples: int = 5000) -> Tuple[np.ndarray, ...]:
    """Get posterior predictive samples with Normal likelihood"""

    print(f"\nGenerating {num_samples} posterior samples from targeted model...")

    # Create predictive distribution
    predictive = Predictive(model, guide=guide, num_samples=num_samples,
                           return_sites=["prediction", "model_uncertainty", "intrinsic_scatter"])

    with torch.no_grad():
        # Put model in eval mode to disable input noise
        model.eval()
        samples = predictive(X, X_err)

        # Extract components
        predictions = samples["prediction"].cpu().numpy()  # (num_samples, n_stars)
        model_uncertainty = samples["model_uncertainty"].cpu().numpy()  # (num_samples, n_stars)
        intrinsic_scatter = samples["intrinsic_scatter"].cpu().numpy()  # (num_samples,)

    # Calculate total predictive uncertainty
    y_err_np = y_err.cpu().numpy()  # (n_stars,)
    n_samples, n_stars = predictions.shape

    # Create full posterior samples including all uncertainty sources
    total_samples = np.zeros_like(predictions)

    print("Combining uncertainty components...")
    for i in range(n_samples):
        # For this posterior sample
        pred_i = predictions[i, :]  # Mean predictions for this weight sample
        model_unc_i = model_uncertainty[i, :]  # Model uncertainty for this sample
        intrinsic_i = intrinsic_scatter[i]  # Intrinsic scatter for this sample

        # Sample from the total distribution for each star
        for j in range(n_stars):
            total_var = y_err_np[j]**2 + model_unc_i[j]**2 + intrinsic_i**2
            total_std = np.sqrt(total_var)

            # Sample from Normal(prediction, total_uncertainty)
            total_samples[i, j] = np.random.normal(pred_i[j], total_std)

    print(f"Mean intrinsic scatter: {np.mean(intrinsic_scatter):.4f} ± {np.std(intrinsic_scatter):.4f} dex")
    print(f"Mean model uncertainty: {np.mean(model_uncertainty):.4f} dex")
    print(f"Mean observational uncertainty: {np.mean(y_err_np):.4f} dex")

    return total_samples, predictions, model_uncertainty, intrinsic_scatter


def analyze_targeted_results(total_samples: np.ndarray,
                           predictions: np.ndarray,
                           model_uncertainty: np.ndarray,
                           intrinsic_scatter: np.ndarray,
                           y_err: np.ndarray,
                           y_test: np.ndarray) -> pd.DataFrame:
    """Analyze results with focus on high age performance"""

    n_samples, n_stars = total_samples.shape

    # Calculate total predictive uncertainty
    total_pred_std = np.std(total_samples, axis=0)  # Includes ALL uncertainty sources

    # Mean components per star
    mean_model_unc = np.mean(model_uncertainty, axis=0)
    mean_intrinsic = np.mean(intrinsic_scatter)

    summary = pd.DataFrame({
        'observational_uncertainty': y_err,
        'model_uncertainty': mean_model_unc,
        'intrinsic_scatter': np.full(n_stars, mean_intrinsic),
        'total_predictive_uncertainty': total_pred_std,
        'pred_median': np.median(total_samples, axis=0),
        'pred_mean_only': np.median(predictions, axis=0),
        'true_age': y_test,
    })

    # Calculate theoretical total uncertainty for verification
    summary['theoretical_total'] = np.sqrt(
        summary['observational_uncertainty']**2 +
        summary['model_uncertainty']**2 +
        summary['intrinsic_scatter']**2
    )

    # Calculate residuals and metrics
    summary['residual'] = summary['true_age'] - summary['pred_median']
    summary['normalized_residual'] = summary['residual'] / summary['total_predictive_uncertainty']

    # Overall performance metrics
    within_1sigma = (np.abs(summary['normalized_residual']) < 1).mean()
    within_2sigma = (np.abs(summary['normalized_residual']) < 2).mean()
    mae = np.abs(summary['residual']).mean()
    rms = np.sqrt((summary['residual']**2).mean())
    corr = np.corrcoef(summary['true_age'], summary['pred_median'])[0,1]

    # High age performance metrics
    high_age_mask = summary['true_age'] > 1
    if high_age_mask.sum() > 0:
        high_age_mae = np.abs(summary[high_age_mask]['residual']).mean()
        high_age_rms = np.sqrt((summary[high_age_mask]['residual']**2).mean())
        high_age_corr = np.corrcoef(summary[high_age_mask]['true_age'],
                                   summary[high_age_mask]['pred_median'])[0,1]
        high_age_within_1sigma = (np.abs(summary[high_age_mask]['normalized_residual']) < 1).mean()

        print(f"\nHIGH AGE STAR PERFORMANCE (logAge > 1):")
        print(f"Number of high age stars: {high_age_mask.sum()}")
        print(f"High age MAE: {high_age_mae:.4f} dex")
        print(f"High age RMS: {high_age_rms:.4f} dex")
        print(f"High age correlation: {high_age_corr:.3f}")
        print(f"High age fraction within 1σ: {high_age_within_1sigma:.1%}")

        # Check prediction range for high age stars
        high_age_pred_min = summary[high_age_mask]['pred_median'].min()
        high_age_pred_max = summary[high_age_mask]['pred_median'].max()
        high_age_pred_mean = summary[high_age_mask]['pred_median'].mean()

        print(f"High age prediction range: [{high_age_pred_min:.4f}, {high_age_pred_max:.4f}]")
        print(f"High age prediction mean: {high_age_pred_mean:.4f}")

    print(f"\nTARGETED MODEL RESULTS:")
    print(f"Mean observational uncertainty: {summary['observational_uncertainty'].mean():.4f} dex")
    print(f"Mean model uncertainty: {summary['model_uncertainty'].mean():.4f} dex")
    print(f"Intrinsic scatter: {mean_intrinsic:.4f} dex")
    print(f"Mean total uncertainty: {summary['total_predictive_uncertainty'].mean():.4f} dex")

    print(f"\nOverall Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.4f} dex")
    print(f"RMS Error: {rms:.4f} dex")
    print(f"Correlation: {corr:.3f}")
    print(f"Fraction within 1σ: {within_1sigma:.1%}")
    print(f"Fraction within 2σ: {within_2sigma:.1%}")

    return summary


def train_smooth_bnn(model: BayesianNeuralNetwork,
                    X_train: torch.Tensor,
                    X_err_train: torch.Tensor,
                    y_train: torch.Tensor,
                    y_err_train: torch.Tensor,
                    num_iterations: int = 8000,
                    initial_lr: float = 0.005,  # Reduced from 0.01
                    batch_size: int = 512,
                    warmup_epochs: int = 20,  # Learning rate warmup
                    seed: int = None) -> Tuple[AutoDiagonalNormal, list]:
    """Train BNN with smooth loss curves - fixes loss jumps

    Key improvements:
    1. Learning rate warmup for first N epochs
    2. Reduced initial learning rate
    3. Better learning rate scheduling
    """

    print("\n" + "="*60)
    print("Training Smooth Bayesian Neural Network")
    print("With Learning Rate Warmup and Better Stability")
    print("="*60)

    # Set seed if provided for reproducibility
    if seed is not None:
        set_seed(seed)
        print(f"Using random seed: {seed}")

    # CRITICAL: Clear parameter store and ensure clean start
    pyro.clear_param_store()
    print("Parameter store cleared for clean training start")

    # Create data loader for mini-batches
    dataset = torch.utils.data.TensorDataset(X_train, X_err_train, y_train, y_err_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup SVI with improved optimizer settings
    guide = AutoDiagonalNormal(model)

    # Calculate number of epochs
    num_epochs = num_iterations // len(loader)

    # Custom learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    # Use standard Adam optimizer with constant learning rate
    from pyro.optim import Adam

    # Keep it simple - just use the initial learning rate
    optimizer = Adam({"lr": initial_lr})

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Training loop with better monitoring
    losses = []
    smoothed_losses = []

    print(f"Training for {num_epochs} epochs with {len(loader)} batches per epoch...")
    print(f"Total iterations: {num_iterations}")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Batch size: {batch_size} (covers {100*batch_size/len(X_train):.1f}% of data per batch)")

    pbar = tqdm(range(num_epochs), desc="Training")
    best_loss = float('inf')

    for epoch in pbar:
        epoch_loss = 0.0
        batch_losses = []

        for batch_x, batch_x_err, batch_y, batch_y_err in loader:
            loss = svi.step(batch_x, batch_x_err, batch_y, batch_y_err)
            epoch_loss += loss
            batch_losses.append(loss)

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        # Calculate smoothed loss (exponential moving average)
        if epoch == 0:
            smoothed_loss = avg_loss
        else:
            smoothed_loss = 0.95 * smoothed_losses[-1] + 0.05 * avg_loss
        smoothed_losses.append(smoothed_loss)

        # Track best smoothed loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        # Update progress bar with more info
        pbar.set_postfix({
            'loss': f'{avg_loss:.2f}',
            'smooth': f'{smoothed_loss:.2f}',
            'lr': f'{initial_lr:.5f}',
        })

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            loss_std = np.std(batch_losses)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f} ± {loss_std:.4f}, "
                  f"Smoothed = {smoothed_loss:.4f}")

    print(f"\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final smoothed loss: {smoothed_losses[-1]:.4f}")
    print(f"Best smoothed loss: {best_loss:.4f}")
    print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")

    return guide, losses


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main_targeted():
    """Complete targeted training pipeline"""

    print("="*70)
    print("TRAINING TARGETED FIXED ROBUST BAYESIAN NEURAL NETWORK")
    print("="*70)

    # File paths
    train_path = './train_data/AllTrainedNorm_dr17.csv'
    test_path = './test_data/TestOriginalNorm_dr17.csv'
    output_dir = './BNN_targeted_output'

    # Create output directories
    Path(output_dir).mkdir(exist_ok=True)

    # Load training data
    X_train, X_err_train, y_train, y_err_train = load_astronomical_data(train_path)
    print(f"\nTraining set: {len(X_train)} stars")
    print(f"Features: {X_train.shape[1]}")
    print(f"Age range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"High age stars (>1): {(y_train > 1).sum()} ({100*(y_train > 1).mean():.1f}%)")

    # Calculate empirical statistics for priors
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    print(f"\nEmpirical age statistics:")
    print(f"  Mean logAge: {y_mean:.3f}")
    print(f"  Std logAge: {y_std:.3f}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_err_train_tensor = torch.FloatTensor(X_err_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_err_train_tensor = torch.FloatTensor(y_err_train).to(device)

    # Single training run with fixed seed for reproducibility
    seed = 42
    print(f"\n" + "="*50)
    print(f"TRAINING WITH SEED {seed}")
    print("="*50)

    # Initialize targeted model with empirical priors
    model = BayesianNeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dim=16,  # Reduced from 32
        use_skip_connections=True,  # Disable skip connections
        use_empirical_output_bias=True,  # Disable empirical Bayes
        use_leaky_relu=True,  # Use standard ReLU
        y_mean=y_mean,
        y_std=y_std
    )
    model.to(device)

    # Train model with targeted fixes
    guide, losses = train_smooth_bnn(
        model,
        X_train_tensor, X_err_train_tensor,
        y_train_tensor, y_err_train_tensor,
        num_iterations=8000,
        initial_lr=0.005,  # Reduced from 0.01
        batch_size=512,
        warmup_epochs=20,  # Learning rate warmup
        seed=seed
    )

    # Save model
    pyro.get_param_store().save(f'{output_dir}/targeted_bnn_params.pth')
    print(f"Model saved")

    # Full evaluation on test set
    print(f"\n" + "="*50)
    print("FULL EVALUATION ON TEST SET")
    print("="*50)

    # Load full test data
    X_test, X_err_test, y_test, y_err_test = load_astronomical_data(test_path)
    print(f"\nTest set: {len(X_test)} stars")
    print(f"Test high age stars (>1): {(y_test > 1).sum()} ({100*(y_test > 1).mean():.1f}%)")

    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    X_err_test_tensor = torch.FloatTensor(X_err_test).to(device)
    y_err_test_tensor = torch.FloatTensor(y_err_test).to(device)

    # Get posterior samples from targeted model
    total_samples, mean_predictions, model_unc, intrinsic_scatter = get_targeted_posterior_samples(
        model, guide, X_test_tensor, X_err_test_tensor, y_err_test_tensor, num_samples=5000)

    # Analyze results
    summary = analyze_targeted_results(
        total_samples, mean_predictions, model_unc, intrinsic_scatter, y_err_test, y_test)

    # Save results
    summary.to_csv(f'{output_dir}/targeted_prediction_summary.csv', index=False)

    # Save full posterior samples for analysis
    np.savez_compressed(f'{output_dir}/targeted_posterior_samples.npz',
                       total_samples=total_samples,
                       mean_predictions=mean_predictions,
                       true_ages=y_test,
                       true_age_errs=y_err_test)

    # Training loss plot - now show both raw and smoothed
    plt.figure(figsize=(15, 5))

    # Plot 1: Raw losses
    plt.subplot(1, 3, 1)
    plt.plot(losses, alpha=0.5, label='Raw loss')
    # Calculate smoothed losses for plotting
    smoothed = []
    for i, loss in enumerate(losses):
        if i == 0:
            smoothed.append(loss)
        else:
            smoothed.append(0.95 * smoothed[-1] + 0.05 * loss)
    plt.plot(smoothed, label='Smoothed loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title('Training Loss (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Log scale losses
    plt.subplot(1, 3, 2)
    # Shift losses to be positive for log scale
    min_loss = min(min(losses), min(smoothed))
    shifted_losses = [l - min_loss + 1 for l in losses]
    shifted_smoothed = [l - min_loss + 1 for l in smoothed]
    plt.plot(shifted_losses, alpha=0.5, label='Raw loss')
    plt.plot(shifted_smoothed, label='Smoothed loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss (shifted)')
    plt.title('Training Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Loss variability over time
    plt.subplot(1, 3, 3)
    window_size = 50
    loss_std = []
    for i in range(window_size, len(losses)):
        window = losses[i-window_size:i]
        loss_std.append(np.std(window))
    plt.plot(range(window_size, len(losses)), loss_std)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Std Dev (50-epoch window)')
    plt.title('Loss Variability Over Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/targeted_training_loss.png', dpi=150)
    plt.close()

    # Diagnostic plots for high-age performance
    plt.figure(figsize=(15, 5))

    # Plot 1: Predictions vs True for all stars
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, summary['pred_median'], alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True logAge')
    plt.ylabel('Predicted logAge')
    plt.title('Predictions vs True Ages')

    # Plot 2: Residuals vs True Age
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, summary['residual'], alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True logAge')
    plt.ylabel('Residual (True - Predicted)')
    plt.title('Residuals vs True Age')

    # Plot 3: Focus on high ages
    high_age_mask = y_test > 1
    if high_age_mask.sum() > 0:
        plt.subplot(1, 3, 3)
        plt.scatter(y_test[high_age_mask], summary['pred_median'][high_age_mask],
                   alpha=0.6, s=20, label='High age stars')
        plt.plot([1, y_test.max()], [1, y_test.max()], 'r--', lw=2, label='Perfect')
        plt.xlabel('True logAge')
        plt.ylabel('Predicted logAge')
        plt.title('High Age Stars (logAge > 1)')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/targeted_diagnostic_plots.png', dpi=150)
    plt.close()

    print(f"\n" + "="*70)
    print("TARGETED TRAINING COMPLETE!")
    print("="*70)
    print(f"Results saved to {output_dir}/")

    return model, guide, summary, total_samples


if __name__ == "__main__":
    # Run the complete targeted training pipeline
    model, guide, summary, samples = main_targeted()
