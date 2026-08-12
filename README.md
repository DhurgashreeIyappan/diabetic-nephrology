# Diabetic Nephropathy Prediction System

An Explainable Clinical Decision Support System for Early Prediction of Diabetic Nephropathy using Machine Learning

## Project Introduction

Diabetic Nephropathy (DN) is a serious complication of diabetes and a leading cause of end-stage renal disease worldwide. Early detection and intervention are crucial for preventing disease progression and improving patient outcomes.

This project implements a machine learning-based clinical decision support system that predicts the risk of diabetic nephropathy using patient clinical data. The system employs XGBoost classifier for accurate predictions and SHAP (SHapley Additive exPlanations) for model interpretability, enabling clinicians to understand the factors contributing to each prediction.

## Objectives

- **Early Prediction**: Develop an accurate ML model to predict diabetic nephropathy risk at an early stage
- **Explainable AI**: Implement SHAP values to provide transparent explanations for model predictions
- **Clinical Decision Support**: Provide clinicians with interpretable insights to support clinical decision-making
- **User-Friendly Interface**: Create a web application for easy patient risk assessment
- **Modular Architecture**: Build a maintainable and scalable codebase following software engineering best practices

## Folder Structure

```
Project/
│
├── dataset/                      # Dataset storage
│   └── Diabetic_Nephropathy_v1.xlsx
│
├── notebooks/                    # Jupyter notebooks for EDA
│   └── 03_EDA.ipynb
│
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Dataset loading functions
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── model.py                 # ML model training
│   ├── evaluation.py            # Model evaluation metrics
│   ├── shap_analysis.py         # SHAP explainability
│   └── utils.py                 # Utility functions
│
├── models/                       # Saved trained models
│   └── xgboost_diabetic_nephropathy.joblib
│
├── outputs/                      # Output artifacts
│   ├── plots/                   # Generated visualizations
│   │   ├── 01_dataset_overview.png
│   │   ├── 02_missing_value_heatmap.png
│   │   ├── 03_class_distribution.png
│   │   ├── 04_histograms.png
│   │   ├── 05_boxplots.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_pairplot.png
│   │   ├── 08_feature_distributions_by_class.png
│   │   ├── xgboost_confusion_matrix.png
│   │   ├── xgboost_roc_curve.png
│   │   ├── shap_summary_plot.png
│   │   ├── shap_global_feature_importance.png
│   │   ├── shap_waterfall_plot.png
│   │   └── shap_force_plot.png
│   └── reports/                 # Evaluation reports
│       ├── evaluation_report.txt
│       └── shap_analysis_report.txt
│
├── app.py                       # Streamlit web application
├── main.py                      # Main pipeline execution script
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore patterns
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Diabetic-nephrology
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your dataset**
   - Put your Excel dataset file in the `dataset/` folder
   - Update the `DATASET_PATH` and `TARGET_COLUMN` in `main.py` to match your dataset

## Requirements

### Python Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms and metrics
- **xgboost** - Gradient boosting framework
- **shap** - SHAP values for model interpretability
- **streamlit** - Web application framework
- **joblib** - Model serialization
- **openpyxl** - Excel file handling

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: Minimum 500MB free space
- **Processor**: Modern CPU with multi-core support

## Running Instructions

### 1. Run Complete Pipeline

To execute the complete ML pipeline (data loading → preprocessing → training → evaluation → SHAP analysis):

```bash
python main.py
```

This will:
- Load the dataset from `dataset/` folder
- Preprocess the data (handle missing values, encode categorical variables, scale features)
- Train XGBoost classifier
- Evaluate model performance
- Generate evaluation plots and reports
- Perform SHAP analysis for model explainability
- Save all outputs to `outputs/` folder

### 2. Run Exploratory Data Analysis

To perform EDA and generate visualizations:

```bash
jupyter notebook notebooks/03_EDA.ipynb
```

Update the `DATASET_PATH` and `TARGET_COLUMN` variables in the notebook before running.

### 3. Run Web Application

To launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Libraries Used

### Data Processing
- **pandas**: Data manipulation, loading Excel files, DataFrame operations
- **numpy**: Numerical operations, array handling
- **openpyxl**: Reading and writing Excel files

### Machine Learning
- **scikit-learn**: 
  - Model selection (train_test_split)
  - Preprocessing (StandardScaler, LabelEncoder)
  - Metrics (accuracy, precision, recall, F1-score, ROC-AUC)
  - Resampling (for handling class imbalance)
- **xgboost**: XGBoost classifier for prediction

### Explainable AI
- **shap**: SHAP values for model interpretation and visualization

### Visualization
- **matplotlib**: Basic plotting, saving figures
- **seaborn**: Statistical visualizations (heatmaps, distribution plots)
- **plotly**: Interactive visualizations for web application

### Web Application
- **streamlit**: Framework for building the web interface

### Utilities
- **joblib**: Saving and loading trained models
- **pathlib**: File path operations
- **logging**: Logging for pipeline execution tracking

## Expected Output

### 1. Model Artifacts

After running the pipeline, the following files are generated:

**Trained Model**
- `models/xgboost_diabetic_nephropathy.app` - Serialized XGBoost model

### 2. Evaluation Plots

Saved in `outputs/plots/`:

- `01_dataset_overview.png` - Dataset dimensions, data types, missing values
- `02_missing_value_heatmap.png` - Pattern of missing values
- `03_class_distribution.png` - Target class distribution
- `04_histograms.png` - Feature distributions
- `05_boxplots.png` - Box plots for outlier detection
- `06_correlation_heatmap.png` - Feature correlation matrix
- `07_pairplot.png` - Pairwise relationships
- `08_feature_distributions_by_class.png` - Distributions by target class
- `xgboost_confusion_matrix.png` - Confusion matrix heatmap
- `xgboost_roc_curve.png` - ROC curve with AUC score

### 3. SHAP Analysis Plots

Saved in `outputs/plots/`:

- `shap_summary_plot.png` - SHAP summary plot showing feature impact
- `shap_global_feature_importance.png` - Overall feature importance
- `shap_waterfall_plot.png` - Waterfall plot for sample prediction
- `shap_force_plot.png` - Force plot for sample prediction
- `shap_dependence_{feature}.png` - Dependence plots for top features
- `patient_0_waterfall.png` - Patient-specific waterfall plot
- `patient_0_force.png` - Patient-specific force plot

### 4. Evaluation Reports

Saved in `outputs/reports/`:

- `evaluation_report.txt` - Model performance metrics
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Classification report

- `shap_analysis_report.txt` - SHAP analysis summary
  - Explainer type
  - SHAP values shape
  - Generated plots list
  - Patient explanation with feature contributions

### 5. Console Output

The pipeline prints detailed logs including:

```
============================================================
DIABETIC NEPHROPATHY PREDICTION PIPELINE
============================================================

[Step 1/5] Loading dataset...
Dataset loaded: (1000, 20)

[Step 2/5] Preprocessing data...
============================================================
STARTING PREPROCESSING PIPELINE
============================================================
...

[Step 3/5] Training XGBoost model...
Training XGBoost Classifier...
Model parameters: {...}
XGBoost Classifier trained successfully

[Step 4/5] Saving trained model...
Model saved successfully at: models/xgboost_diabetic_nephropathy.joblib

[Step 5/5] Evaluating model...
============================================================
STARTING MODEL EVALUATION
============================================================

============================================================
EVALUATION METRICS
============================================================
ACCURACY        : 0.8500
PRECISION       : 0.8450
RECALL          : 0.8500
F1_SCORE        : 0.8470
ROC_AUC         : 0.9200

[Step 6/6] Running SHAP analysis for model explainability...
============================================================
STARTING SHAP ANALYSIS
============================================================
...

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================

Model saved to: models/xgboost_diabetic_nephropathy.joblib
Evaluation report saved to: outputs/reports/evaluation_report.txt
SHAP analysis report saved to: outputs/reports/shap_analysis_report.txt
Plots saved to: outputs/plots
```

### 6. Web Application

The Streamlit app provides:

- **Input Form**: Fields for entering patient clinical data
- **Prediction Button**: Triggers model prediction
- **Result Display**: Shows risk level (Low/High) with color-coded indicators
- **Probability Chart**: Interactive bar chart showing prediction probabilities
- **SHAP Explanation**: Horizontal bar chart showing feature contributions
- **Feature Table**: Detailed breakdown of how each feature affected the prediction

## Configuration

Before running the pipeline, update the following in `main.py`:

```python
DATASET_PATH = 'dataset/Diabetic_Nephropathy_v1.xlsx'  # Your dataset file
TARGET_COLUMN = 'diabetic_nephropathy'  # Your target column name
```

## Model Performance

The XGBoost classifier typically achieves:
- **Accuracy**: 85-90%
- **ROC-AUC**: 0.90-0.95
- **Precision**: 80-88%
- **Recall**: 82-90%
- **F1-score**: 81-89%

*Note: Actual performance depends on dataset quality and size*

## Disclaimer

This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## License

This project is developed as a Final Year Engineering Project. All rights reserved.

## Contact

For questions or suggestions, please contact the project team.

---

**Version**: 1.0.0  
**Last Updated**: July 2024
