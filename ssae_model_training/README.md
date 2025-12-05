# Student Success Analytics & Early Intervention System (SSAES) - Model Training

A complete machine learning pipeline for predicting student performance and identifying at-risk students for early intervention.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your dataset:**
   - Place your CSV file in `data/demo/` folder
   - The notebook expects a file with columns like: `student_id`, `final_marks`, `pass_fail`, `attendance_rate`, etc.

3. **Run the training notebook:**
   ```bash
   jupyter notebook notebooks/01_model_training.ipynb
   ```

## Project Structure

```
ssae_model_training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/demo/                   # Place your CSV datasets here
├── notebooks/01_model_training.ipynb  # Main training pipeline
├── src/utils.py                 # Helper functions
├── models/                      # Saved model artifacts
└── reports/figures/             # Generated plots and results
```

## Features

- **Regression Models:** Predict final marks (continuous target)
- **Classification Models:** Predict pass/fail outcomes (binary target)
- **Model Comparison:** Linear Regression, Logistic Regression, Random Forest, XGBoost
- **Hyperparameter Tuning:** GridSearchCV for optimal parameters
- **Model Explainability:** SHAP analysis for feature importance
- **Automated Evaluation:** Comprehensive metrics and visualizations

## Output

- Trained models saved to `models/`
- Evaluation plots and metrics in `reports/figures/`
- Model comparison tables for easy selection

## Next Steps

- Deploy best models to Django web application
- Implement real-time alert system for at-risk students
- Add more sophisticated feature engineering
- Integrate with student information systems