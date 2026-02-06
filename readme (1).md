# AB Testing Pro v2.0

### By Harsha

A powerful, beginner-friendly platform for analyzing A/B test experiments. Built with Streamlit, it combines statistical testing and machine learning to help you understand what works, why it works, and who it works best for — no data science degree required.

---

## Features

### Simple A/B Test
For anyone who just wants a quick answer. Enter your numbers directly — no file upload needed. Results are explained in plain English with a clear winner/loser verdict and confidence level.

### Advanced A/B Test Analysis
Upload a CSV or Excel file with raw experiment data. The platform automatically detects your group column, outcome column, and features, then runs the appropriate statistical test:
- **Welch's t-test** — for comparing averages (e.g., revenue per user)
- **Chi-square test** — for comparing rates (e.g., conversion rates)
- **Mann-Whitney U test** — for non-normal or skewed data

### ML Model Studio
Train machine learning models on your experiment data to go beyond simple averages:
- **Predictive modeling** — predict outcomes like conversions, revenue, or engagement
- **Uplift modeling** — identify which users benefit the most from your treatment
- Trains multiple algorithms (Gradient Boosting, Random Forest, Logistic Regression / Ridge) and picks the best one
- Feature importance analysis to understand what drives results

### Predictions
Score new data using your trained models. Upload fresh user data, get predictions, and download the results — no need to re-run the experiment.

### Sample Datasets
Six built-in demo datasets across different industries so you can explore the tool without uploading anything:
- Tech product conversions
- Ecommerce cart values
- Marketing email clicks
- Gaming session lengths
- Finance signup rates
- Healthcare treatment outcomes

Plus a custom data generator for creating large-scale synthetic datasets.

---

## Quick Start

### Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ab-testing-pro
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:5000`.

### On Replit

Click **Run** — the app starts automatically on port 5000.

---

## What Kind of Data Works?

This app is built for **A/B testing data** — any dataset where you split people into groups and measured something.

### Your data should have:
| Column Type | Description | Examples |
|-------------|-------------|----------|
| **Group column** | Which group each user was in | "control" vs "treatment", "A" vs "B", "old" vs "new" |
| **Outcome column** | The thing you're measuring | Converted (yes/no), revenue ($), time on page, clicks |
| **Feature columns** (optional) | Extra details about users | Age, device, location, plan type, signup date |

### Works great for:
- Website A/B tests (which design got more clicks?)
- Pricing experiments (did the discount increase purchases?)
- Email campaigns (which subject line got more opens?)
- App feature rollouts (did the new feature improve engagement?)
- Ad creative testing (which ad drove more conversions?)
- Clinical trials (did the treatment improve outcomes?)

### Not designed for:
- Datasets without a group/comparison structure
- Time series forecasting
- Image or text classification
- Data with no measurable outcome

The groups don't need to be labeled "A" and "B" — the app auto-detects them, even if they're called "exposed" vs "unexposed" or "variant_1" vs "variant_2".

---

## Project Structure

```
ab-testing-pro/
├── app.py                     # Main application (UI, routing, visualization)
├── statistical_tests.py       # Statistical testing module
├── ml_engine.py               # Machine learning engine
├── sample_data.py             # Sample dataset generator
├── enhanced_data_generator.py # Large-scale synthetic data generator
├── requirements.txt           # Python dependencies
├── readme.md                  # This file
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

### Module Breakdown

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with sidebar navigation, data upload, Plotly visualizations, and page routing |
| `statistical_tests.py` | Pure statistical tests (z-test, t-test, Mann-Whitney, chi-square) with auto-recommendation |
| `ml_engine.py` | Multi-model ML engine with auto column detection, cross-validation, uplift modeling, and feature importance |
| `sample_data.py` | Six pre-built industry demo datasets |
| `enhanced_data_generator.py` | Generates 100k+ row synthetic datasets with realistic patterns across multiple domains |

---

## Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.11** | Runtime |
| **Streamlit** | Web framework and UI |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computation |
| **SciPy** | Statistical tests |
| **scikit-learn** | Machine learning models and evaluation |
| **Plotly** | Interactive charts and visualizations |
| **Joblib** | Model serialization |
| **OpenPyXL** | Excel file support |
| **PyArrow** | Parquet file support |

---

## How It Works

1. **Upload or enter data** — Use the Simple A/B Test page for quick number entry, or upload a CSV/Excel file on the Advanced page.

2. **Auto-detection** — The app inspects your data and automatically identifies the group column, outcome column, and feature columns based on data types, unique values, and naming patterns.

3. **Statistical testing** — The appropriate test is selected based on your data characteristics. You get a p-value, confidence level, and a plain-English explanation of what the result means.

4. **ML modeling** (optional) — Train multiple model types on your data. The engine compares Gradient Boosting, Random Forest, and linear models, then reports accuracy, feature importance, and (for uplift models) which user segments benefit most from treatment.

5. **Predictions** (optional) — Load a trained model and score new data to predict outcomes without running another experiment.

---

## Design Philosophy

- **Two-tier UX**: Simple mode for beginners (just enter numbers), Advanced mode for power users (upload full datasets)
- **Plain English results**: Every statistical result comes with a human-readable explanation — no jargon
- **Zero configuration**: Auto-detection means users don't have to manually map columns or choose test types
- **Multi-model approach**: Trains several algorithms and compares them, so users get the best model without needing ML expertise
- **Self-contained**: No external APIs, databases, or authentication required — everything runs in-browser

---

## License

This project is provided as-is for educational and analytical use.
