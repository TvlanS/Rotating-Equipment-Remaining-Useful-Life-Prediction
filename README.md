# Rotating Equipment Prediction - Remaining Useful Life (RUL) Estimation

## Project Overview

This project implements predictive maintenance models for estimating the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS FD001 dataset. The goal is to predict how many operational cycles remain before engine failure by analyzing sensor data and operational settings. The project compares six different modeling approaches (both traditional machine learning and deep learning) and incorporates Explainable AI (XAI) techniques using SHAP for model interpretability.

**Key Features:**
- Comprehensive comparison of 6 models: Linear Regression, SVR, Random Forest, CNN, LSTM, and TSMixer
- Temporal windowing for sequence modeling
- Early RUL capping to stabilize targets
- Model-specific feature scaling
- SHAP-based model interpretability
- Hyperparameter tuning for conventional models

## Dataset

The project uses the **NASA CMAPSS FD001** dataset from NASA's Prognostics Center of Excellence:

- **Training data**: `train_FD001.txt` - 100 engine run-to-failure trajectories
- **Test data**: `RUL_FD001.txt` - True RUL values for test engines
- **Data structure**: 26 columns including:
  - Unit number
  - Time (in cycles)
  - 3 operational settings
  - 21 sensor measurements
- **Conditions**: Single operating condition (Sea Level)
- **Fault mode**: HPC (High Pressure Compressor) degradation

**Dataset Sources:**
- Research Paper: [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf)
- Dataset: [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Installation & Dependencies

### Python Environment
- Python 3.10+ recommended

### Required Packages
Install the following dependencies:

```bash
pip install numpy==1.26.4
pip install pandas
pip install scikit-learn
pip install tensorflow
pip install shap
pip install matplotlib
pip install joblib
```

All dependencies are listed in the notebook and will be installed automatically when running the notebook cells.

## Project Structure

```
86.0 Rotating Equipment Prediction/
├── 0.0 Dataset/                    # NASA CMAPSS FD001 dataset
│   ├── readme.txt                 # Dataset description
│   ├── train_FD001.txt            # Training data (100 engines)
│   └── RUL_FD001.txt              # Test RUL values
├── 1.0 Pickle Files/              # Saved scalers
│   ├── MinMax.pkl                 # MinMax scaler for conventional models
│   └── target_scaler_cnn.pkl      # Target scaler for CNN
├── 2.0 Slides/                    # Project presentation
│   └── Project Slide.pdf          # Summary slides
├── 3_1_Final_Version 3.ipynb      # Main Jupyter notebook
└── README.md                      # This file
```

## Usage

### Running the Notebook
1. Open `3_1_Final_Version 3.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially from top to bottom
3. The notebook includes:
   - Data loading and preprocessing
   - Model training and evaluation
   - Hyperparameter tuning for conventional models
   - SHAP analysis for model interpretability

### Key Functions
- `process_targets()`: Early RUL capping and target processing
- `process_input_data_with_targets()`: Window-based temporal processing
- `process_test_data()`: Test data processing

### Model Training
The notebook implements six models with consistent interfaces:
1. **Linear Regression** (baseline)
2. **Support Vector Regression (SVR)** with hyperparameter tuning
3. **Random Forest Regressor** with hyperparameter tuning
4. **CNN** (Convolutional Neural Network)
5. **LSTM** (Long Short-Term Memory)
6. **TSMixer** (Transformer-based model)

## Model Performance

| Model | MAE | R² | RMSE | S-score |
|-------|-----|----|------|---------|
| **LSTM** | 10.92 | 0.876 | 14.02 | - |
| **CNN** | 10.41 | 0.874 | 14.14 | - |
| **TSMixer** | 10.39 | 0.875 | 14.05 | - |
| **Random Forest** | 13.21 | 0.783 | 18.53 | - |
| **SVR** | 14.80 | 0.735 | 20.49 | - |
| **Linear Regression** | 13.94 | 0.817 | 17.04 | - |

*Note: Performance metrics are from the notebook evaluation. Lower S-score/MAE/RMSE and higher R² indicate better performance.*

## Key Findings

1. **Deep Learning Superiority**: LSTM, CNN, and TSMixer outperformed traditional machine learning models in capturing temporal degradation patterns.

2. **Best Performing Model**: LSTM achieved the best overall performance with MAE: 10.92, R²: 0.876, and RMSE: 14.02.

3. **Temporal Modeling**: Window-based sequence processing (window size = 25 for deep learning models) significantly improved performance for time-series data.

4. **Explainability**: SHAP analysis revealed that sensor measurements 2, 3, 4, 7, 11, 12, 15, and 21 were most influential for RUL predictions, aligning with physical degradation patterns.

5. **Early RUL Capping**: Capping RUL values at 125 cycles stabilized target distributions and improved model training.

6. **Model Complexity Trade-off**: While deep learning models performed better, they require more computational resources and training time compared to traditional models.

## Explainable AI (XAI)

The project incorporates SHAP (SHapley Additive exPlanations) to interpret model predictions:
- Identifies which sensor measurements most influence RUL predictions
- Provides transparency into "black-box" deep learning models
- Validates that models learn physically meaningful degradation patterns
- Enhances trust in predictive maintenance systems

## Technical Details

### Preprocessing Pipeline
1. **Early RUL Capping**: Limits maximum RUL to 125 cycles to prevent target distribution skew
2. **Temporal Windowing**: Creates sequences of 25 time steps for deep learning models
3. **Model-Specific Scaling**:
   - Conventional models: MinMax scaling
   - Deep learning models: Custom scaling approaches
4. **Feature Selection**: Based on sensor variability and correlation analysis

### Model Architectures
- **CNN**: 1D convolutional layers for temporal feature extraction
- **LSTM**: Bidirectional LSTM layers for sequence modeling
- **TSMixer**: Transformer-based architecture for time-series forecasting
- **Conventional Models**: Scikit-learn implementations with hyperparameter tuning

## References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation". *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08)*, Denver, CO.

2. NASA Prognostics Center of Excellence. "C-MAPSS: Commercial Modular Aero-Propulsion System Simulation". https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

3. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions". *Advances in Neural Information Processing Systems*.

## License

This project is for educational and research purposes. The dataset is provided by NASA and subject to their terms of use.

## Acknowledgements

- NASA Prognostics Center of Excellence for the CMAPSS dataset
- Contributors to the open-source libraries used in this project
- Academic researchers in predictive maintenance and prognostic health management