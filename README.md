# Fraud Detection ML Pipeline

A production-grade machine learning pipeline for detecting fraudulent transactions in imbalanced financial datasets.

## 🎯 Project Purpose

This project provides a comprehensive, modular framework for building and deploying fraud detection models. It's designed to handle the challenges of imbalanced financial datasets with robust preprocessing, multiple model training, comprehensive evaluation, and model explainability.

## 📁 Project Structure

```
fraud-detection/
├── data/                 # Raw and processed data
│   ├── raw/             # Original datasets
│   └── processed/       # Cleaned and engineered data
├── notebooks/           # Jupyter notebooks for EDA and experimentation
│   └── eda.ipynb       # Exploratory Data Analysis notebook
├── src/                 # Core source code
│   ├── __init__.py
│   ├── config.py        # Configuration settings
│   ├── data_loader.py   # Data loading and validation
│   ├── preprocess.py    # Feature engineering and preprocessing
│   ├── explainability.py # Model interpretation and SHAP analysis
│   ├── utils.py         # Utility functions and logging
│   └── models/          # Model training and evaluation
│       ├── __init__.py
│       ├── train.py     # Model training pipeline
│       └── evaluate.py  # Model evaluation and metrics
├── tests/               # Pytest-based test suite
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_preprocess.py
├── logs/                # Runtime logs
├── .github/workflows/   # CI/CD workflows
│   └── ci.yml          # GitHub Actions CI pipeline
├── .gitignore          # Git ignore patterns
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── run.py             # Main execution script
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
# Load and preprocess data
from src.data_loader import load_sample_data
from src.preprocess import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

# Load sample data
df = load_sample_data()

# Preprocess data
preprocessor = DataPreprocessor()
df_processed = preprocessor.fit_transform(df, 'fraud')

# Train models
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(df_processed, 'fraud')
models = trainer.train_all_models(X_train, y_train)

# Evaluate models
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models, X_test, y_test)
```

### 3. Run the Pipeline

```bash
# Run the complete pipeline
python run.py

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_data_loader.py
```

## 🔧 Key Features

### Data Management
- **Robust Data Loading**: Support for CSV, Parquet, and other formats
- **Data Validation**: Comprehensive quality checks and validation
- **Sample Data Generation**: Synthetic data for development and testing

### Preprocessing
- **Feature Engineering**: Time-based, amount-based, and interaction features
- **Handling Imbalanced Data**: SMOTE, undersampling, and combined techniques
- **Feature Scaling**: Standard, Robust, and MinMax scaling options
- **Feature Selection**: Automatic selection of most important features

### Model Training
- **Multiple Algorithms**: Random Forest, Logistic Regression, XGBoost
- **Cross-Validation**: Stratified k-fold cross-validation
- **Hyperparameter Configuration**: Pre-configured optimal parameters
- **Model Persistence**: Save and load trained models

### Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Visualization**: Confusion matrices, ROC curves, precision-recall curves
- **Model Comparison**: Side-by-side comparison of multiple models
- **Detailed Reports**: Automated evaluation reports with recommendations

### Explainability
- **SHAP Analysis**: Model interpretation using SHAP values
- **Feature Importance**: Multiple methods for feature importance calculation
- **Prediction Explanations**: Individual prediction explanations
- **Visual Reports**: Automated explainability reports

## 📊 Model Performance

The pipeline is designed to handle imbalanced datasets typical in fraud detection:

- **Class Imbalance**: Built-in handling of 1-5% fraud rates
- **Evaluation Focus**: Emphasis on precision, recall, and F1-score
- **Business Metrics**: Cost-sensitive evaluation metrics
- **Threshold Optimization**: Automatic threshold tuning for business needs

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test category
pytest tests/test_data_loader.py
pytest tests/test_preprocess.py
```

## 📈 Continuous Integration

The project includes GitHub Actions CI/CD pipeline that:

- Runs on Python 3.10+
- Installs dependencies
- Runs all tests
- Generates coverage reports
- Validates code quality

## 🔍 Exploratory Data Analysis

Use the provided Jupyter notebook for comprehensive EDA:

```bash
# Start Jupyter
jupyter notebook notebooks/eda.ipynb
```

The EDA notebook covers:
- Data overview and quality assessment
- Target variable analysis
- Feature analysis and correlations
- Time-based patterns
- Amount analysis
- Merchant category analysis
- Feature engineering insights

## 📝 Configuration

All configuration is centralized in `src/config.py`:

- **Data paths**: Raw and processed data directories
- **Model parameters**: Hyperparameters for all algorithms
- **Evaluation metrics**: Metrics to compute and report
- **Logging settings**: Log levels and formats
- **Feature engineering**: Scaling and encoding methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Check the documentation in the code
- Review the EDA notebook for examples
- Open an issue on GitHub

## 🔮 Future Enhancements

- Real-time prediction API
- Model monitoring and drift detection
- Advanced ensemble methods
- Deep learning models
- Automated hyperparameter tuning
- Model deployment tools 