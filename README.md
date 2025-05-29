# Robust Bayesian Neural Network for Stellar Age Prediction

An implementation of Bayesian Neural Networks (BNNs) for predicting stellar ages from spectro data, with uncertainty quantification and diagnostic tools.

## 🌟 Features

- **Robust Bayesian Neural Network**: Implements a targeted BNN with empirical Bayes priors and LeakyReLU activation
- **Uncertainty Quantification**: Provides full posterior distributions for age predictions with uncertainty estimates
- **Comprehensive Diagnostics**: Multiple visualization tools for model performance analysis
- **Data Handling**: Supports astronomical survey data with proper error propagation
- **Reproducible Results**: Built-in seed management for consistent results

## 📁 Project Structure

```
bingo-modern/
├── train_bnn.py                    # Main BNN training script
├── diagnose_pdfs.py                # Posterior PDF visualization
├── plot_age_predictions.py         # Age prediction plotting
├── plot_predicted_vs_true.py       # Model performance comparison
├── plot_with_uncertainties.py      # Uncertainty visualization
├── train_data/                     # Training datasets
│   └── AllTrainedNorm_dr17.csv
├── test_data/                      # Test datasets
│   └── TestOriginalNorm_dr17.csv
└── BNN_targeted_output/            # Model outputs and results
    ├── targeted_bnn_params.pth
    ├── targeted_posterior_samples.npz
    ├── targeted_prediction_summary.csv
    └── *.png (diagnostic plots)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy pandas pyro-ppl matplotlib seaborn scipy tqdm
```

### Training the Model

```bash
python train_bnn.py
```

This will:
- Load and preprocess the astronomical data
- Train the Bayesian Neural Network
- Generate posterior samples
- Save model parameters and predictions
- Create diagnostic plots

### Generating Visualizations

```bash
# Create age prediction plots
python plot_age_predictions.py

# Generate predicted vs true comparisons
python plot_predicted_vs_true.py

# Visualize uncertainties
python plot_with_uncertainties.py

# Generate posterior PDF diagnostics
python diagnose_pdfs.py
```

## 🔬 Model Architecture

The BNN features:

- **3-layer neural network** with LeakyReLU activation
- **Empirical Bayes priors** based on data statistics
- **Automatic variational inference** using PyTorch and Pyro
- **Robust loss tracking** with reproducibility features
- **Edge case handling** for extreme age predictions

### Key Improvements

1. **Proper parameter store clearing** for consistent training
2. **Better loss tracking** and monitoring
3. **LeakyReLU activation** to prevent dying neurons
4. **Complex architecture** for improved high-age predictions
5. **Comprehensive uncertainty quantification**

## 📊 Output Files

### Model Files
- `targeted_bnn_params.pth`: Trained model parameters
- `targeted_posterior_samples.npz`: Posterior samples for inference

### Data Files
- `targeted_prediction_summary.csv`: Predictions with uncertainties

### Visualizations
- `targeted_training_loss.png`: Training loss curves
- `targeted_diagnostic_plots.png`: Model diagnostic plots
- `predictions_with_uncertainties.png`: Uncertainty visualizations
- `diagnostic_pdfs_*.png`: Posterior distribution plots

## 🎯 Use Cases

- **Stellar Age Estimation**: Predict ages of stars from photometric and spectroscopic data
- **Uncertainty Analysis**: Quantify prediction confidence for astronomical surveys
- **Model Validation**: Compare predictions against known stellar populations
- **Survey Planning**: Identify targets requiring follow-up observations

## 📈 Performance Metrics

The model provides:
- **Point estimates** with mean and median predictions
- **Uncertainty intervals** (68% and 95% credible intervals)
- **Prediction quality metrics** (MAE, RMSE, correlation)
- **Calibration diagnostics** for uncertainty validation

## 🛠️ Customization

### Data Format
The model expects CSV files with normalized astronomical features. Modify the data loading functions in `train_bnn.py` for different input formats.

### Model Configuration
Key parameters can be adjusted in `train_bnn.py`:
- Network architecture (hidden dimensions, layers)
- Training parameters (learning rate, epochs)
- Prior distributions
- Uncertainty thresholds

### Visualization Options
Each plotting script includes configuration sections for:
- Output folders and filenames
- Plot styling and colors
- Data filtering criteria
- Figure dimensions

## 📚 Dependencies

- **PyTorch**: Neural network framework
- **Pyro**: Probabilistic programming for Bayesian inference
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **SciPy**: Statistical functions
- **tqdm**: Progress bars

## 🤝 Contributing

This project implements cutting-edge Bayesian methods for astronomical applications. Contributions are welcome for:
- Model architecture improvements
- Additional diagnostic tools
- Performance optimizations
- Documentation enhancements

## 📄 License

Open source - feel free to use and modify for research and educational purposes.

## 🔗 Related Work

This implementation addresses common issues in astronomical age prediction including:
- High-dimensional feature spaces
- Heteroscedastic uncertainties
- Extreme value predictions
- Model calibration and validation

For more details on the methodology, see the extensive comments in `train_bnn.py`.
