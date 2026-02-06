# AB Testing Pro

## Overview

AB Testing Pro is a Streamlit-based web application for running A/B test analysis with statistical testing and machine learning capabilities. Users can upload datasets (or use built-in sample datasets), auto-detect relevant columns, run statistical tests (t-test, chi-square, Mann-Whitney), train ML models for predictive and uplift modeling, and generate predictions on new data. The app serves as an all-in-one platform for experiment analysis across multiple industry domains (tech, ecommerce, marketing, gaming, finance, healthcare).

## User Preferences

Preferred communication style: Simple, everyday language.

- Use a modern dark UI theme with gradient cards and styled metric displays
- Sidebar navigation layout for page switching
- Auto-detect dataset columns (group, target, numeric, categorical) so users don't have to manually configure everything
- Provide sample datasets so users can explore the tool without uploading their own data
- Keep the interface intuitive — minimize required configuration steps

## System Architecture

### Application Structure

The app follows a modular single-page-app pattern using Streamlit with sidebar navigation. Each module handles a specific concern:

- **`app.py`** — Main Streamlit entry point. Handles UI layout, page routing via sidebar, data upload, visualization (Plotly charts), and orchestrates calls to the statistical and ML modules. Runs on port 5000.
- **`statistical_tests.py`** — Pure statistical testing logic. Implements two-proportion z-test, Welch's t-test, Mann-Whitney U, and chi-square tests. Returns structured `TestResult` dataclass objects. Also includes a `recommend_test` method to auto-select the appropriate test based on data characteristics.
- **`ml_engine.py`** — Universal ML engine (`UniversalMLEngine` class). Handles auto-detection of column types, supports both classification and regression tasks, trains multiple model types (Gradient Boosting, Random Forest, Logistic Regression/Ridge), computes feature importance, and supports uplift modeling. Uses scikit-learn throughout.
- **`sample_data.py`** — `SampleDataGenerator` class providing 6 pre-built sample datasets (tech conversions, ecommerce cart values, marketing clicks, gaming sessions, etc.) for demo/exploration purposes.
- **`enhanced_data_generator.py`** — `EnhancedDataGenerator` for creating large-scale (100k+) multi-domain synthetic training datasets with complex realistic patterns. Used for robust model training scenarios.

### Design Decisions

**Framework: Streamlit** — Chosen for rapid prototyping of data-focused web apps. Provides built-in widgets, session state management, and easy deployment. Trade-off: limited customization compared to full web frameworks, but ideal for analytical tools.

**ML Approach: Multi-model ensemble** — The ML engine trains multiple model types (GradientBoosting, RandomForest, linear models) and compares them, rather than relying on a single algorithm. This gives users better coverage across different data characteristics without requiring ML expertise.

**Auto-detection pattern** — The `auto_detect_columns` method inspects data types, unique value counts, and column names to automatically suggest group columns, target columns, and feature types. This reduces friction for non-technical users.

**Statistical testing module** — Kept separate from ML to maintain clean separation of concerns. Statistical tests are pure functions with no side effects, making them easy to test and extend.

### Configuration

- **Streamlit config**: `.streamlit/config.toml` — configured for port 5000, headless mode, CORS disabled
- **Model serialization**: Uses joblib for saving/loading trained models

## External Dependencies

### Python Packages (from requirements.txt)

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework, serves the UI on port 5000 |
| `pandas` | Data manipulation and DataFrame operations |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models (GradientBoosting, RandomForest, LogisticRegression, Ridge), preprocessing (StandardScaler, LabelEncoder), evaluation metrics, cross-validation |
| `scipy` | Statistical tests (t-test, chi-square, Mann-Whitney U, z-test) |
| `joblib` | Model serialization (save/load trained models) |
| `plotly` | Interactive data visualizations (charts, graphs) |
| `openpyxl` | Excel file reading/writing support |
| `python-dateutil` | Date parsing utilities |
| `pyarrow` | Parquet file support and optimized data processing |

### External Services

- **No external APIs or databases** — The application is fully self-contained. All data processing happens in-memory. Users upload data via the Streamlit file uploader, and results are displayed in the browser or downloaded as files.
- **No authentication** — The app is designed for single-user or team use without login requirements.
- **No persistent storage** — Data and models exist only in the Streamlit session state during runtime. Model files can be exported via joblib but there's no built-in database.