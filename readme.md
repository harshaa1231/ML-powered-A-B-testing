# AB Testing Pro - v2.0
### By Harsha

## Overview
A Streamlit-based AB testing platform with statistical analysis and universal ML-powered modeling. Users can upload any dataset, train ML models (predictive and uplift), run statistical tests, and generate predictions.

## Project Architecture
- **app.py** - Main Streamlit application with modern sidebar navigation UI
- **ml_engine.py** - Universal ML engine that handles any dataset (auto-detection, training, prediction, uplift modeling)
- **statistical_tests.py** - Statistical testing module (t-test, chi-square, Mann-Whitney)
- **sample_data.py** - Sample dataset generator for demo purposes
- **enhanced_data_generator.py** - Enhanced multi-domain data generation utilities
- **requirements.txt** - Python dependencies
- **readme.md** - Project documentation

## Tech Stack
- Python 3.11
- Streamlit (web framework)
- Pandas, NumPy (data processing)
- SciPy, scikit-learn (statistical tests & ML)
- Plotly (visualization)
- Joblib (model serialization)

## Key Features
- **AB Test Analysis**: Upload data, auto-detect columns, run statistical tests
- **ML Model Studio**: Train predictive models or uplift models on any dataset
- **Predictions**: Score new data with trained models, download results
- **Sample Datasets**: 6 industry-specific demo datasets + custom data generation
- **Universal ML Engine**: Auto-detects column types, handles classification/regression, feature importance

## Configuration
- Streamlit config: `.streamlit/config.toml` (port 5000, headless, CORS disabled)
- Dependencies: `requirements.txt`

## Running
- Workflow: `streamlit run app.py` on port 5000
- Deployment: autoscale target with same command

## User Preferences
- Modern dark UI theme with gradient cards
- Sidebar navigation layout
- Auto-detection of dataset columns for ease of use
