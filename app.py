# AB Testing Pro v2.0
# Author: Harsha
# ML-Powered A/B Testing & Analysis Platform

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px

from statistical_tests import StatisticalTester
from sample_data import SampleDataGenerator
from ml_engine import UniversalMLEngine

st.set_page_config(
    page_title="AB Testing Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 12px;
    }
    .metric-card h3 {
        font-size: 14px;
        color: #94a3b8 !important;
        margin: 0 0 8px 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #f1f5f9 !important;
        margin: 0;
    }
    .metric-card .sub {
        font-size: 13px;
        color: #64748b !important;
        margin: 4px 0 0 0;
    }
    .success-card {
        border-left-color: #10b981;
    }
    .warning-card {
        border-left-color: #f59e0b;
    }
    .danger-card {
        border-left-color: #ef4444;
    }
    .info-card {
        border-left-color: #8b5cf6;
    }
    .hero-section {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .hero-section h1 {
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
    }
    .hero-section p {
        color: #94a3b8;
        font-size: 1.1em;
        margin: 0;
    }
    .section-header {
        font-size: 1.5em;
        font-weight: 700;
        color: #f1f5f9;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    }
    .result-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
    }
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 28px;
        border: 1px solid rgba(59, 130, 246, 0.15);
        text-align: center;
        height: 100%;
        min-height: 180px;
    }
    .feature-card .icon {
        font-size: 40px;
        margin-bottom: 14px;
    }
    .feature-card h4 {
        color: #e2e8f0 !important;
        margin: 0 0 10px 0;
        font-size: 18px;
    }
    .feature-card p {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
        margin: 0;
    }
    .step-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 28px;
        border: 1px solid rgba(59, 130, 246, 0.1);
        text-align: center;
        position: relative;
    }
    .step-card .step-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 14px;
    }
    .step-card h4 {
        color: #e2e8f0 !important;
        margin: 0 0 10px 0;
        font-size: 16px;
    }
    .step-card p {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
        margin: 0;
    }
    .help-box {
        background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
        border-radius: 14px;
        padding: 22px 26px;
        border-left: 4px solid #60a5fa;
        margin: 10px 0 18px 0;
    }
    .help-box p {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.7;
        margin: 0;
    }
    .help-box strong {
        color: #f1f5f9;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'ml_engine' not in st.session_state:
    st.session_state.ml_engine = None
if 'trained_model_info' not in st.session_state:
    st.session_state.trained_model_info = None
if 'saved_models' not in st.session_state:
    st.session_state.saved_models = []
if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = None

import re
def sanitize_filename(name):
    safe = re.sub(r'[^A-Za-z0-9_\- ]', '', name).strip().replace(' ', '_')
    return safe if safe else 'model'


def render_metric_card(title, value, subtitle="", card_class=""):
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <h3>{title}</h3>
        <p class="value">{value}</p>
        <p class="sub">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def create_chart(x_data, y_data, title, x_label, y_label, colors=None, chart_type='bar'):
    if chart_type == 'bar':
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_data, y=y_data,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.4f}" if isinstance(v, float) else str(v) for v in y_data],
            textposition='outside',
            textfont=dict(size=11)
        ))
    elif chart_type == 'scatter':
        fig = px.scatter(x=x_data, y=y_data, title=title)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#e2e8f0')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#94a3b8'),
        margin=dict(t=50, b=50, l=50, r=30),
        xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
        yaxis=dict(gridcolor='rgba(148,163,184,0.1)')
    )
    return fig


def run_ab_analysis(df, group_col, metric_col, test_type, domain="general"):
    try:
        group_vals = df[group_col].astype(str).str.strip().str.lower()
        control_data = df[group_vals.isin(['control', '0', 'a', 'baseline', 'unexposed'])][metric_col].values
        treatment_data = df[group_vals.isin(['treatment', '1', 'b', 'variant', 'exposed', 'test'])][metric_col].values

        if len(control_data) == 0 or len(treatment_data) == 0:
            unique_groups = group_vals.unique()
            if len(unique_groups) >= 2:
                control_data = df[group_vals == unique_groups[0]][metric_col].values
                treatment_data = df[group_vals == unique_groups[1]][metric_col].values
            else:
                st.error("Could not identify control and treatment groups.")
                return None

        tester = StatisticalTester()

        if test_type == "ttest":
            results = tester.independent_ttest(control_data, treatment_data)
        elif test_type == "chi_square":
            results = tester.chi_square_test(control_data, treatment_data)
        elif test_type == "mann_whitney":
            results = tester.mann_whitney_u_test(control_data, treatment_data)
        else:
            metric_type = "categorical" if df[metric_col].nunique() <= 2 else "continuous"
            recommended = tester.recommend_test(control_data, treatment_data, metric_type)
            if recommended == "chi_square":
                results = tester.chi_square_test(control_data, treatment_data)
            elif recommended == "mann_whitney":
                results = tester.mann_whitney_u_test(control_data, treatment_data)
            else:
                results = tester.independent_ttest(control_data, treatment_data)

        results['domain'] = domain
        results['metric'] = metric_col
        results['test_type'] = test_type
        results['n_control'] = len(control_data)
        results['n_treatment'] = len(treatment_data)
        results['ml_enabled'] = False

        return results
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 2.5em; margin-bottom: 8px;'>🔬</div>
        <h2 style='margin: 0; font-size: 1.3em; background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>AB Testing Pro</h2>
        <p style='font-size: 0.85em; color: #64748b !important; margin: 4px 0 0 0;'>ML-Powered Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    pages = ["Home", "Learn A/B Testing", "Simple A/B Test", "AB Test Analysis", "ML Model Studio", "Predictions", "Sample Datasets"]
    nav_target = st.session_state.pop('_nav_target', None)
    if nav_target and nav_target in pages:
        st.session_state['nav_page'] = nav_target
    page = st.radio(
        "Navigation",
        pages,
        key="nav_page",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**Quick Stats**")
    st.caption(f"Tests Run: {len(st.session_state.predictions)}")
    model_status = "Trained" if st.session_state.ml_engine and st.session_state.ml_engine.is_trained else "Not trained"
    st.caption(f"ML Model: {model_status}")

    st.divider()
    st.caption("v2.0 - Universal ML Engine")


if page == "Home":
    st.markdown("""
    <div class="hero-section" style="padding: 60px 40px;">
        <h1 style="font-size: 3.2em; margin-bottom: 16px;">AB Testing Pro</h1>
        <p style="font-size: 1.25em; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            Find out what works and what doesn't — powered by data, explained in plain English.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    st.markdown("""
    <div class="metric-card" style="border-left-color: #60a5fa; padding: 30px;">
        <h3 style="text-transform: none !important; font-size: 18px !important; color: #60a5fa !important; margin-bottom: 14px;">What is A/B Testing?</h3>
        <p class="sub" style="font-size: 15px !important; line-height: 1.9; color: #cbd5e1 !important;">
            Imagine you're trying to decide between two versions of something — two website designs, two email subject lines,
            two pricing options, or two ad creatives.
            <br><br>
            Instead of guessing which one is better, you <strong style="color: #f1f5f9;">show Version A to one group of people and Version B to another</strong>. 
            Then you compare the results to see which one actually performed better.
            <br><br>
            <strong style="color: #f1f5f9;">That's it. That's A/B testing.</strong>
            <br><br>
            This tool does the math for you. It tells you whether the difference you're seeing is real or just random luck
            — and explains everything in language anyone can understand. No technical background needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("### How It Works")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="step-card">
            <div class="step-number">1</div>
            <h4>You ran an experiment</h4>
            <p>You showed two versions of something to different groups of people and collected the results.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="step-card">
            <div class="step-number">2</div>
            <h4>Enter your data</h4>
            <p>Type in your numbers directly, or upload a data file (CSV or Excel). The tool handles the rest.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="step-card">
            <div class="step-number">3</div>
            <h4>We crunch the numbers</h4>
            <p>The tool runs the right statistical test automatically and checks if the difference is real or just luck.</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="step-card">
            <div class="step-number">4</div>
            <h4>Get a clear answer</h4>
            <p>You'll see a plain-English verdict: who won, by how much, and how confident we are in the result.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("### Where to Start")
    st.caption("Pick the option that best fits your situation:")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="feature-card">
            <div class="icon">✨</div>
            <h4>New to A/B testing?</h4>
            <p>Go to <strong>Simple A/B Test</strong> in the sidebar. Just type in your numbers — how many people saw each version and what happened. You'll get a plain-English answer in seconds. No files, no technical knowledge needed.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="feature-card">
            <div class="icon">📊</div>
            <h4>Have a data file?</h4>
            <p>Go to <strong>AB Test Analysis</strong> to upload your CSV or Excel file. The tool automatically figures out which columns are your groups and results, picks the right test, and gives you detailed visualizations.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""<div class="feature-card">
            <div class="icon">🤖</div>
            <h4>Want deeper insights?</h4>
            <p>Use the <strong>ML Model Studio</strong> to train a machine learning model on your experiment data. It finds hidden patterns and can predict which users benefit most from your changes — going beyond simple averages.</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="feature-card">
            <div class="icon">🧪</div>
            <h4>Just exploring?</h4>
            <p>Check out <strong>Sample Datasets</strong> to play with ready-made examples from tech, ecommerce, marketing, gaming, finance, and healthcare. Great for learning how the tool works before using your own data.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("")

    with st.expander("Common questions from first-time users"):
        st.markdown("""
**Do I need to know statistics or data science?**
Not at all. The tool does all the math behind the scenes and explains every result in plain language. You just need to know what you tested and what happened.

**What kind of results can I test?**
Anything where you compared two options:
- Click rates, signup rates, purchase rates (yes/no outcomes)
- Revenue, time spent, scores, ratings (number outcomes)
- Email open rates, ad performance, feature engagement — the list goes on

**How many people do I need in my test?**
More is better, but a few hundred per group is usually enough to detect meaningful differences. The tool will tell you if your sample is too small to draw conclusions.

**What's the difference between Simple A/B Test and AB Test Analysis?**
- **Simple A/B Test** — You type in summary numbers (e.g., "1000 people saw A, 50 clicked"). Quick and easy.
- **AB Test Analysis** — You upload a full data file with one row per person. More detailed, with charts and deeper analysis.

**What does "statistically significant" mean?**
It means we're confident the difference between A and B is real — not just random luck. Think of it like flipping a coin: if you get 6 heads out of 10 flips, that could be chance. But 600 heads out of 1000? Something is clearly going on. We use the same kind of logic to test your experiment results.

**Can I use this for free?**
Yes — there's nothing to install, no account needed, and no limits on how many tests you can run.
        """)

    st.markdown("")

    if st.session_state.predictions:
        st.markdown("")
        st.markdown('<div class="section-header">Your Recent Results</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Tests", len(st.session_state.predictions), "experiments analyzed")
        with col2:
            sig = sum(1 for p in st.session_state.predictions if p.get('is_significant', False))
            render_metric_card("Significant", sig, f"{sig/len(st.session_state.predictions)*100:.0f}% of tests", "success-card")
        with col3:
            avg_up = np.mean([p.get('uplift_percentage', 0) for p in st.session_state.predictions])
            render_metric_card("Avg Uplift", f"{avg_up:.2f}%", "across all tests", "warning-card" if avg_up > 0 else "danger-card")
        with col4:
            avg_p = np.mean([p.get('p_value', 1) for p in st.session_state.predictions])
            render_metric_card("Avg P-Value", f"{avg_p:.4f}", "lower = more confident", "info-card")


elif page == "Learn A/B Testing":
    st.markdown('<div class="section-header">Learn A/B Testing</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="border-left-color: #a78bfa; padding: 30px;">
        <h3 style="text-transform: none !important; font-size: 18px !important; color: #a78bfa !important;">Your crash course in A/B testing</h3>
        <p class="sub" style="font-size: 15px !important; line-height: 1.7;">
            Read through the lessons below — each one takes about 2 minutes. By the end, you'll understand 
            everything you need to run and interpret A/B tests confidently. No prior knowledge required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    st.markdown("### Lesson 1: What is A/B Testing?")
    st.markdown("""
    Imagine you own a lemonade stand. You want to know if a **new sign** brings in more customers than the old one.

    So you try both: on Monday, Wednesday, Friday you use the old sign. On Tuesday, Thursday, Saturday you use the new one. 
    Then you compare how many cups you sold on each day.

    That's A/B testing. **Version A** is your current way, **Version B** is the new thing you're trying. 
    You compare them to see which one works better.

    Companies like Google, Amazon, Netflix, and Airbnb run thousands of A/B tests every year. 
    Every button color, headline, pricing change, and feature is tested this way.
    """)

    st.markdown("")
    st.markdown("### Lesson 2: Why Can't I Just Look at the Numbers?")
    st.markdown("""
    Let's say Version A got 100 clicks and Version B got 110 clicks. B is clearly better, right?

    **Not necessarily.** That difference could just be random chance — like how some days are busier than others 
    for no particular reason.

    A/B testing uses math to answer one critical question: 
    **"Is this difference real, or could it just be luck?"**

    That's what makes it more reliable than gut instinct. Your gut might say B is better. 
    The math might say "we can't be sure yet — you need more data."
    """)

    st.markdown("""
    <div class="help-box">
        <p><strong>Key concept: Statistical Significance</strong><br>
        When we say a result is "statistically significant," it means we've done the math and we're at least 
        95% confident the difference is real — not just luck. This is the gold standard for making decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### Lesson 3: The Numbers You'll See")
    st.markdown("""
    When you run a test in this app, you'll see a few key numbers. Here's what they all mean:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #60a5fa;">
            <h3 style="text-transform: none !important; font-size: 15px !important; color: #60a5fa !important;">Conversion Rate</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                The percentage of people who did what you wanted.<br>
                <strong style="color: #f1f5f9;">Example:</strong> 50 signups out of 1,000 visitors = 5% conversion rate.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <h3 style="text-transform: none !important; font-size: 15px !important; color: #f59e0b !important;">P-Value</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                The probability your result is just random noise.<br>
                <strong style="color: #f1f5f9;">Low p-value (under 0.05)</strong> = the result is probably real.<br>
                <strong style="color: #f1f5f9;">High p-value (over 0.05)</strong> = could be luck.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #10b981;">
            <h3 style="text-transform: none !important; font-size: 15px !important; color: #10b981 !important;">Lift / Difference</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                How much better (or worse) B is compared to A, shown as a percentage.<br>
                <strong style="color: #f1f5f9;">Example:</strong> A = 5%, B = 6%. That's a 20% lift (because 6 is 20% more than 5).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="border-left-color: #ef4444;">
            <h3 style="text-transform: none !important; font-size: 15px !important; color: #ef4444 !important;">Confidence Level</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                How sure we are the result is real (the opposite of p-value).<br>
                <strong style="color: #f1f5f9;">95%+ confidence</strong> = safe to act on.<br>
                <strong style="color: #f1f5f9;">Under 95%</strong> = need more data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### Lesson 4: How to Run a Good A/B Test")
    st.markdown("""
    Not all A/B tests are created equal. Here are the rules the pros follow:

    **1. Test one thing at a time**
    If you change the button color AND the headline AND the price, you won't know which change made the difference. Change one thing per test.

    **2. Split your audience randomly**
    Don't show Version A to returning customers and Version B to new visitors. The groups should be random so they're comparable.

    **3. Decide your success metric before you start**
    Are you measuring clicks? Purchases? Revenue? Time on page? Pick one main metric and stick with it.

    **4. Run the test long enough**
    Don't check the results after 2 hours and make a decision. Wait until you have enough data (usually at least a few hundred people per group). This app will tell you if your sample is too small.

    **5. Don't peek and stop early**
    If you check results every hour and stop the moment B looks better, you might be fooled by a random streak. Let the test run to completion.
    """)

    st.markdown("")
    st.markdown("### Lesson 5: Common Mistakes (and How to Avoid Them)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card danger-card">
            <h3 style="text-transform: none !important; font-size: 15px !important;">Mistake: Too small a sample</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                Testing with 20 people per group won't give you reliable results. 
                You need hundreds or thousands depending on the size of the difference you're looking for.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card danger-card">
            <h3 style="text-transform: none !important; font-size: 15px !important;">Mistake: Testing too many things at once</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                If you changed 5 things between A and B, and B won, which change actually helped? You have no idea. Test one change at a time.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card danger-card">
            <h3 style="text-transform: none !important; font-size: 15px !important;">Mistake: Stopping too early</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                Don't stop the test just because B looks good after an hour. Early results are unreliable. 
                Wait until you have enough data for a confident conclusion.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card danger-card">
            <h3 style="text-transform: none !important; font-size: 15px !important;">Mistake: Ignoring "no difference" results</h3>
            <p class="sub" style="font-size: 14px !important; line-height: 1.6;">
                If your test shows no significant difference, that's still valuable. It means your change doesn't matter — so don't waste resources on it. Move on to the next idea.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### Lesson 6: Your Learning Path in This App")
    st.markdown("""
    Now that you understand the basics, here's how to put it into practice:
    """)

    st.markdown("""
    <div class="step-card" style="margin-bottom: 16px;">
        <div class="step-number">1</div>
        <h4>Start with Simple A/B Test</h4>
        <p>Type in your experiment numbers and get a plain-English answer. Perfect for your first test. No files needed.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card" style="margin-bottom: 16px;">
        <div class="step-number">2</div>
        <h4>Try AB Test Analysis with a sample dataset</h4>
        <p>Go to Sample Datasets, download one, then upload it to AB Test Analysis. See how the tool auto-detects columns and picks the right test.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card" style="margin-bottom: 16px;">
        <div class="step-number">3</div>
        <h4>Upload your own data</h4>
        <p>Once you're comfortable, bring your own experiment data. The tool will guide you through every step.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card" style="margin-bottom: 16px;">
        <div class="step-number">4</div>
        <h4>Explore ML Model Studio (optional)</h4>
        <p>When you're ready for advanced insights, train a machine learning model to predict who'll convert and find hidden patterns in your data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("I'm ready! Take me to Simple A/B Test", type="primary"):
        st.session_state['_nav_target'] = 'Simple A/B Test'
        st.rerun()


elif page == "Simple A/B Test":
    st.markdown('<div class="section-header">Simple A/B Test</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="border-left-color: #60a5fa;">
        <h3 style="text-transform: none !important; font-size: 16px !important;">The easiest way to test your ideas</h3>
        <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
            You tried two different versions of something (Version A and Version B). Now you want to know:
            <strong style="color: #f1f5f9;">did one actually perform better, or was it just luck?</strong>
            Enter your numbers below and we'll tell you in plain English — no math or statistics knowledge needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    simple_mode = st.radio(
        "What kind of result did you measure?",
        ["Something people click or sign up for (yes/no outcome)", "Something you can measure with numbers (like revenue or time spent)"],
        help="Pick the one that best describes your experiment"
    )

    with st.expander("Not sure which one to pick? Here are some examples"):
        st.markdown("""
**Yes/No outcomes** (first option) — use this when you're counting how many people did something:
- Did they click a button? (yes or no)
- Did they sign up? (yes or no)
- Did they buy something? (yes or no)
- Did they open the email? (yes or no)

**Number outcomes** (second option) — use this when you're measuring *how much* of something:
- How much money did they spend? ($45.20)
- How many minutes did they stay on the page? (3.5 min)
- What rating did they give? (4.2 out of 5)
- How many items did they add to cart? (2.3 items)
        """)

    st.markdown("---")

    if "yes/no" in simple_mode:
        st.markdown("#### Enter your results")
        st.caption("For example: You showed two versions of a button. How many people saw each version, and how many clicked?")
        st.info("**Quick tip:** The \"conversion rate\" is just the percentage of people who took action. If 1000 people saw your page and 50 signed up, your conversion rate is 5%. We calculate this for you automatically.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Version A** (your original)")
            visitors_a = st.number_input("How many people saw Version A?", min_value=1, value=1000, step=100, key="simple_va")
            conversions_a = st.number_input("How many of them took action (clicked, signed up, etc)?", min_value=0, value=50, step=10, key="simple_ca")
        with col2:
            st.markdown("**Version B** (your new version)")
            visitors_b = st.number_input("How many people saw Version B?", min_value=1, value=1000, step=100, key="simple_vb")
            conversions_b = st.number_input("How many of them took action?", min_value=0, value=65, step=10, key="simple_cb")

        if conversions_a > visitors_a:
            st.warning("Version A: Actions can't be more than the number of people who saw it.")
        elif conversions_b > visitors_b:
            st.warning("Version B: Actions can't be more than the number of people who saw it.")
        elif st.button("Tell me the result", type="primary", width='stretch'):
            rate_a = conversions_a / visitors_a
            rate_b = conversions_b / visitors_b
            lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0

            from scipy import stats as sp_stats
            contingency = np.array([[conversions_a, visitors_a - conversions_a],
                                    [conversions_b, visitors_b - conversions_b]])
            try:
                _, p_value, _, _ = sp_stats.chi2_contingency(contingency)
            except:
                p_value = 1.0

            is_significant = p_value < 0.05

            st.markdown("---")
            st.markdown("### Your Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Version A Rate", f"{rate_a*100:.2f}%", f"{conversions_a} out of {visitors_a}")
            with col2:
                render_metric_card("Version B Rate", f"{rate_b*100:.2f}%", f"{conversions_b} out of {visitors_b}")
            with col3:
                lift_class = "success-card" if lift > 0 else "danger-card"
                render_metric_card("Difference", f"{lift:+.2f}%", "B compared to A", lift_class)

            st.markdown("")

            if is_significant and lift > 0:
                st.markdown(f"""
                <div class="metric-card success-card" style="text-align: center;">
                    <h3 style="text-transform: none !important; font-size: 18px !important; color: #10b981 !important;">Version B is the winner!</h3>
                    <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
                        Version B performed <strong style="color: #f1f5f9;">{abs(lift):.1f}% better</strong> than Version A,
                        and we're confident this isn't just random chance.
                        There's only a <strong style="color: #f1f5f9;">{p_value*100:.2f}%</strong> probability this result is a fluke.
                        <br><br>You can go ahead and use Version B.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif is_significant and lift < 0:
                st.markdown(f"""
                <div class="metric-card danger-card" style="text-align: center;">
                    <h3 style="text-transform: none !important; font-size: 18px !important; color: #ef4444 !important;">Version A is better. Keep it.</h3>
                    <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
                        Version B actually performed <strong style="color: #f1f5f9;">{abs(lift):.1f}% worse</strong> than Version A.
                        This result is statistically reliable — it's not just luck.
                        <br><br>Stick with Version A.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card warning-card" style="text-align: center;">
                    <h3 style="text-transform: none !important; font-size: 18px !important; color: #f59e0b !important;">No clear winner yet</h3>
                    <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
                        The difference between A and B is <strong style="color: #f1f5f9;">{abs(lift):.1f}%</strong>,
                        but we can't be sure this isn't just random variation.
                        <br><br>Try running your test longer or with more people to get a clearer answer.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            confidence = (1 - p_value) * 100
            st.markdown(f"""
            <div class="metric-card info-card">
                <h3 style="text-transform: none !important;">What does this mean?</h3>
                <p class="sub" style="font-size: 14px !important; line-height: 1.7;">
                    We're <strong style="color: #f1f5f9;">{confidence:.1f}% confident</strong> in this result.
                    Generally, you want at least 95% confidence before making a decision.
                    {'You have enough confidence to act on this.' if is_significant else 'You need more data or a bigger difference to be sure.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.predictions.append({
                'metric': 'simple_ab_test',
                'domain': 'simple',
                'test_type': 'chi_square',
                'p_value': float(p_value),
                'effect_size': float(abs(rate_b - rate_a)),
                'uplift_percentage': float(lift),
                'is_significant': bool(is_significant),
                'n_control': int(visitors_a),
                'n_treatment': int(visitors_b),
                'test_name': 'Simple A/B Test (Yes/No)',
            })

    else:
        st.markdown("#### Enter your results")
        st.caption("For example: You tested two pricing pages and tracked how much each group spent.")
        st.info("**Quick tip:** \"Average\" means the typical value across all people. \"Standard deviation\" shows how spread out the numbers are — if you don't know it, the default value is fine for most cases.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Version A** (your original)")
            count_a = st.number_input("How many people in Version A?", min_value=2, value=500, step=50, key="simple_na")
            avg_a = st.number_input("What was the average result for A?", value=45.0, step=1.0, format="%.2f", key="simple_avg_a")
            std_a = st.number_input("Standard deviation for A (if you know it, otherwise leave as is)", value=10.0, min_value=0.01, step=1.0, format="%.2f", key="simple_std_a")
        with col2:
            st.markdown("**Version B** (your new version)")
            count_b = st.number_input("How many people in Version B?", min_value=2, value=500, step=50, key="simple_nb")
            avg_b = st.number_input("What was the average result for B?", value=48.0, step=1.0, format="%.2f", key="simple_avg_b")
            std_b = st.number_input("Standard deviation for B", value=10.0, min_value=0.01, step=1.0, format="%.2f", key="simple_std_b")

        if st.button("Tell me the result", type="primary", width='stretch', key="simple_numeric_run"):
            from scipy import stats as sp_stats
            t_stat, p_value = sp_stats.ttest_ind_from_stats(
                mean1=avg_a, std1=std_a, nobs1=count_a,
                mean2=avg_b, std2=std_b, nobs2=count_b
            )
            if np.isnan(p_value):
                p_value = 1.0
            is_significant = p_value < 0.05

            diff = avg_b - avg_a
            lift = ((avg_b - avg_a) / abs(avg_a) * 100) if avg_a != 0 else 0

            st.markdown("---")
            st.markdown("### Your Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Version A Average", f"{avg_a:.2f}", f"{count_a} people")
            with col2:
                render_metric_card("Version B Average", f"{avg_b:.2f}", f"{count_b} people")
            with col3:
                lift_class = "success-card" if diff > 0 else "danger-card"
                render_metric_card("Difference", f"{diff:+.2f}", f"{lift:+.1f}% change", lift_class)

            st.markdown("")

            if is_significant and diff > 0:
                st.markdown(f"""
                <div class="metric-card success-card" style="text-align: center;">
                    <h3 style="text-transform: none !important; font-size: 18px !important; color: #10b981 !important;">Version B gives better results!</h3>
                    <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
                        Version B's average is <strong style="color: #f1f5f9;">{abs(diff):.2f} higher</strong> ({abs(lift):.1f}% better) than Version A.
                        We're confident this is a real improvement, not random noise.
                        <br><br>You can go ahead and use Version B.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif is_significant and diff < 0:
                st.markdown(f"""
                <div class="metric-card danger-card" style="text-align: center;">
                    <h3 style="text-transform: none !important; font-size: 18px !important; color: #ef4444 !important;">Version A is better. Keep it.</h3>
                    <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
                        Version B's average is <strong style="color: #f1f5f9;">{abs(diff):.2f} lower</strong> ({abs(lift):.1f}% worse) than Version A.
                        This is a real difference, not just luck.
                        <br><br>Stick with Version A.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card warning-card" style="text-align: center;">
                    <h3 style="text-transform: none !important; font-size: 18px !important; color: #f59e0b !important;">No clear winner yet</h3>
                    <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
                        The difference of <strong style="color: #f1f5f9;">{abs(diff):.2f}</strong> between A and B
                        could just be random variation.
                        <br><br>Try testing with more people or run the experiment longer.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            confidence = (1 - p_value) * 100
            st.markdown(f"""
            <div class="metric-card info-card">
                <h3 style="text-transform: none !important;">What does this mean?</h3>
                <p class="sub" style="font-size: 14px !important; line-height: 1.7;">
                    We're <strong style="color: #f1f5f9;">{confidence:.1f}% confident</strong> in this result.
                    Generally, you want at least 95% confidence before making a decision.
                    {'You have enough confidence to act on this.' if is_significant else 'You need more data or a bigger difference to be sure.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.predictions.append({
                'metric': 'simple_ab_test',
                'domain': 'simple',
                'test_type': 'ttest',
                'p_value': float(p_value),
                'effect_size': float(abs(diff)),
                'uplift_percentage': float(lift),
                'is_significant': bool(is_significant),
                'n_control': int(count_a),
                'n_treatment': int(count_b),
                'test_name': 'Simple A/B Test (Numeric)',
            })

    st.markdown("")
    st.markdown("---")
    st.markdown("### Learn While You Test")
    st.caption("Expand any section below to learn a concept. By the time you've read them all, you'll understand A/B testing like a pro.")

    with st.expander("What exactly is A/B testing?"):
        st.markdown("""
It's like a science experiment for your business. You show one group of people **Version A** (the original) and another group **Version B** (the new version), then compare which one does better.

**Real-world example:** An online store wants to know if a red "Buy Now" button gets more clicks than a blue one. They show the red button to 5,000 random visitors and the blue button to another 5,000. Then they compare how many people clicked each one.

That's it. You just did A/B testing.
        """)

    with st.expander("What's a conversion rate and why does it matter?"):
        st.markdown("""
A **conversion rate** is the percentage of people who did what you wanted them to do.

**Formula:** Conversions / Total visitors x 100 = Conversion rate

**Example:** 
- 1,000 people visited your page
- 50 of them signed up
- Your conversion rate = 50 / 1,000 x 100 = **5%**

Even a small improvement in conversion rate can mean big business results. Going from 5% to 6% might sound small, but that's a **20% increase** in signups!
        """)

    with st.expander("What does 'statistically significant' mean?"):
        st.markdown("""
It means we're confident the difference you're seeing is **real** — not just random luck.

**Think of it like this:** If you flip a coin 10 times and get 6 heads, is the coin rigged? Probably not — that could easily happen by chance. But if you flip it 1,000 times and get 600 heads? Something is definitely going on.

The same logic applies to your A/B test. We use math to figure out whether the difference between A and B is big enough (and based on enough people) to be trusted.

**The rule of thumb:** We look for **95% confidence** — meaning there's less than a 5% chance the result is a fluke. When you see "statistically significant," that bar has been cleared.
        """)

    with st.expander("What's a p-value? (explained simply)"):
        st.markdown("""
The **p-value** answers one question: *"If there were actually NO difference between A and B, how likely would I be to see results this extreme?"*

- **p-value = 0.03** means there's a 3% chance the result is just noise. That's low — so the result is probably real.
- **p-value = 0.50** means there's a 50% chance it's just noise. That's high — you can't trust this result.

**The magic number is 0.05** (5%). If the p-value is below 0.05, we call it "statistically significant."

Don't worry about the exact number — we tell you the verdict in plain English every time.
        """)

    with st.expander("How many people do I need in my test?"):
        st.markdown("""
More people = more reliable results. Here's a rough guide:

| What you're testing | Minimum per group | Ideal per group |
|---|---|---|
| Big differences (>20% lift) | 100-200 | 500+ |
| Medium differences (5-20%) | 500-1,000 | 2,000+ |
| Small differences (<5%) | 2,000-5,000 | 10,000+ |

**Why?** Small differences are harder to spot. Imagine trying to tell if a coin is slightly biased — you'd need to flip it thousands of times. But if it always lands heads, even 10 flips would tell you.

**Pro tip:** If your test says "no clear winner," try running it with more people before giving up. The difference might be real but your sample was too small to detect it.
        """)

    with st.expander("What should I do after getting my results?"):
        st.markdown("""
**If Version B won (statistically significant):**
1. Roll out Version B to everyone
2. Document what you learned (this builds institutional knowledge)
3. Think about what else you could test — the best companies test constantly

**If Version A won (statistically significant):**
1. Keep Version A — your change didn't work this time
2. Analyze why: Was the change too subtle? Wrong audience? Bad timing?
3. Try a different approach and test again

**If there's no clear winner:**
1. Don't panic — inconclusive results are common and still valuable
2. Consider running the test longer to get more data
3. Try a bigger, bolder change — small tweaks are harder to detect
4. Check if your sample size was big enough (see "How many people do I need?")

**Golden rule:** Never stop testing. Even the best companies run hundreds of A/B tests per year. Each one teaches you something.
        """)

    with st.expander("Ready for the next level? When to use the Advanced pages"):
        st.markdown("""
You've mastered the basics! Here's when to level up:

**Use AB Test Analysis when:**
- You have a full dataset (CSV/Excel) with one row per person
- You want charts and visualizations alongside your results
- You want the tool to automatically pick the best statistical test

**Use ML Model Studio when:**
- You want to understand *why* people converted (not just how many)
- You want to predict who will convert in the future
- You want to find which user segments respond best to your changes

**Use Predictions when:**
- You've trained a model and want to score new people
- You want to prioritize who to target with your campaign

Think of it as a progression: Simple A/B Test (beginner) -> AB Test Analysis (intermediate) -> ML Studio (advanced) -> Predictions (expert)
        """)


elif page == "AB Test Analysis":
    st.markdown('<div class="section-header">AB Test Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="help-box">
        <p>Upload your experiment data and we'll tell you if the difference between your groups is real or just random chance.
        Your file should have <strong>one row per person/observation</strong>, with a column showing which group they were in (e.g., "control" vs "treatment") 
        and a column showing the result (e.g., whether they converted, how much they spent, etc.).</p>
    </div>
    """, unsafe_allow_html=True)

    upload_tab, config_tab = st.tabs(["Upload Data", "Results"])

    with upload_tab:
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.current_df = df

                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric_card("Rows", f"{df.shape[0]:,}", "data points")
                with col2:
                    render_metric_card("Columns", f"{df.shape[1]}", "features")
                with col3:
                    render_metric_card("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB", "dataset size")

                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(15), width='stretch', height=300)

                with st.expander("Column Info"):
                    info_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.notna().sum().values,
                        'Unique': df.nunique().values,
                        'Sample': [str(df[c].iloc[0]) if len(df) > 0 else '' for c in df.columns]
                    })
                    st.dataframe(info_df, width='stretch', hide_index=True)

                st.markdown("---")
                st.subheader("Configure Your Test")
                st.caption("We've auto-detected the most likely columns. Adjust if needed — hover over any option for more details.")

                engine = UniversalMLEngine()
                detection = engine.auto_detect_columns(df)

                col1, col2 = st.columns(2)
                with col1:
                    default_group = detection['potential_group_cols'][0] if detection['potential_group_cols'] else df.columns[0]
                    default_idx = list(df.columns).index(default_group) if default_group in df.columns else 0
                    group_col = st.selectbox("Group Column", df.columns, index=default_idx,
                                            help="Which column tells us what group each person was in? Look for columns with values like 'control/treatment', 'A/B', 'old/new', etc.")
                with col2:
                    default_target = detection['potential_target_cols'][0] if detection['potential_target_cols'] else df.columns[-1]
                    default_idx = list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1
                    metric_col = st.selectbox("Outcome Column", df.columns, index=default_idx,
                                             help="What did you measure? This is the result you care about — like whether someone converted (0/1), how much they spent, or how long they stayed.")

                col1, col2 = st.columns(2)
                with col1:
                    test_type = st.selectbox("Test Type", ["auto", "ttest", "chi_square", "mann_whitney"],
                                           help="Leave on 'auto' and the tool picks the right test for your data. T-test compares averages, Chi-square compares rates (yes/no), Mann-Whitney works when data is skewed.")
                with col2:
                    domain = st.selectbox("Industry Context", ["general", "tech", "ecommerce", "marketing", "gaming", "finance", "healthcare"],
                                         help="Optional — tells the system what kind of experiment this is so it can tailor the language in results.")

                if st.button("Run Analysis", type="primary", width='stretch'):
                    with st.spinner("Running statistical analysis..."):
                        results = run_ab_analysis(df, group_col, metric_col, test_type, domain)
                        if results:
                            st.session_state.predictions.append(results)
                            st.success("Analysis complete! Check the Results tab.")
                            st.balloons()

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with config_tab:
        if st.session_state.predictions:
            st.markdown("""
            <div class="help-box">
                <p><strong>Reading your results:</strong> Each test below shows whether the difference between your groups is real or just random noise. 
                Look at the <strong>verdict badge</strong> (green = real difference, red = not enough evidence) and the <strong>uplift %</strong> to see how much better or worse Version B performed.</p>
            </div>
            """, unsafe_allow_html=True)

            for i, pred in enumerate(reversed(st.session_state.predictions)):
                with st.expander(f"Test {len(st.session_state.predictions) - i}: {pred.get('metric', 'N/A')} ({pred.get('domain', '')})", expanded=(i == 0)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        p_val = pred.get('p_value', 1)
                        sig_class = "success-card" if p_val < 0.05 else "danger-card"
                        render_metric_card("P-Value", f"{p_val:.6f}", "lower = more confident", sig_class)
                    with col2:
                        render_metric_card("Effect Size", f"{pred.get('effect_size', 0):.4f}", pred.get('test_name', ''))
                    with col3:
                        up = pred.get('uplift_percentage', 0)
                        up_class = "success-card" if up > 0 else "danger-card"
                        render_metric_card("Uplift", f"{up:.2f}%", "B compared to A", up_class)
                    with col4:
                        sig = pred.get('is_significant', False)
                        badge = '<span class="result-badge badge-success">REAL DIFFERENCE</span>' if sig else '<span class="result-badge badge-danger">NOT ENOUGH EVIDENCE</span>'
                        st.markdown(f"<div style='padding-top: 30px; text-align: center;'>{badge}</div>", unsafe_allow_html=True)

                    if 'n_control' in pred:
                        st.caption(f"Control group: {pred['n_control']:,} people | Treatment group: {pred['n_treatment']:,} people")

                    confidence = (1 - pred.get('p_value', 1)) * 100
                    if sig:
                        if up > 0:
                            st.success(f"Version B performed {abs(up):.1f}% better than Version A. We're {confidence:.1f}% confident this is a real improvement.")
                            st.markdown("""
                            <div class="help-box" style="margin-top: 8px;">
                                <p><strong>What to do next:</strong> You can confidently roll out Version B. Consider documenting what you changed and why it worked — this builds your team's knowledge base. Then think about what to test next.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Version B performed {abs(up):.1f}% worse than Version A. We're {confidence:.1f}% confident this is a real decline. Stick with A.")
                            st.markdown("""
                            <div class="help-box" style="margin-top: 8px;">
                                <p><strong>What to do next:</strong> Keep Version A. Analyze why the change didn't work — was it the wrong direction, or was the change too subtle? Try a different approach and test again.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"The difference of {abs(up):.1f}% could be random chance. We're only {confidence:.1f}% confident (need 95%+). Try with more data.")
                        total_n = pred.get('n_control', 0) + pred.get('n_treatment', 0)
                        advice = "Consider running the test longer to collect more data, or try a bolder change that creates a bigger difference." if total_n > 200 else "Your sample size is quite small. Try to get at least a few hundred people per group for more reliable results."
                        st.markdown(f"""
                        <div class="help-box" style="margin-top: 8px;">
                            <p><strong>What to do next:</strong> Don't give up — inconclusive results are normal and still valuable. {advice}</p>
                        </div>
                        """, unsafe_allow_html=True)

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                metrics = [p.get('metric', f"Test {i}") for i, p in enumerate(st.session_state.predictions)]
                pvalues = [p.get('p_value', 0) for p in st.session_state.predictions]
                colors = ['#10b981' if p < 0.05 else '#ef4444' for p in pvalues]
                fig = create_chart(metrics, pvalues, "P-Values by Test", "Metric", "P-Value", colors)
                fig.add_hline(y=0.05, line_dash="dash", line_color="#f59e0b",
                             annotation_text="Significance Threshold (0.05)")
                st.plotly_chart(fig, width='stretch')

            with col2:
                uplifts = [p.get('uplift_percentage', 0) for p in st.session_state.predictions]
                colors = ['#10b981' if u > 0 else '#ef4444' for u in uplifts]
                fig = create_chart(metrics, uplifts, "Uplift % by Test", "Metric", "Uplift %", colors)
                fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
                st.plotly_chart(fig, width='stretch')

            csv_data = pd.DataFrame(st.session_state.predictions).to_csv(index=False)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download Results CSV", csv_data, "ab_test_results.csv", "text/csv", width='stretch')
            with col2:
                if st.button("Clear All Results", width='stretch'):
                    st.session_state.predictions = []
                    st.rerun()
        else:
            st.info("No results yet. Upload data and run an analysis in the Upload Data tab.")


elif page == "ML Model Studio":
    st.markdown('<div class="section-header">ML Model Studio (Advanced)</div>', unsafe_allow_html=True)
    st.caption("Train machine learning models on your data. The system automatically figures out the best approach — no ML expertise needed.")

    with st.expander("Quick Guide: ML Terms Explained", expanded=False):
        st.markdown("""
**Classification** — The model predicts a **category** (yes/no, pass/fail, buy/don't buy). Use this when your outcome is a label, not a number.
Example: *Will this user convert?* → Yes or No.

**Regression** — The model predicts a **number** (revenue, time spent, score). Use this when your outcome is a quantity.
Example: *How much will this user spend?* → $42.50.

**Auto** — Let the system decide. It looks at your target column and picks classification or regression automatically based on the data.

**Test Size** — The percentage of your data set aside to check how well the model works. A test size of 0.2 means 20% of your data is reserved for testing, and 80% is used for training. Higher test size = more rigorous check, but less data to learn from.

**Uplift Model** — A special model that answers: *"Who benefits most from the treatment?"* Instead of just predicting outcomes, it compares treatment vs. control to find users who respond best to your intervention. Great for targeting the right audience.

**Feature Importance** — After training, the model tells you which columns (features) had the biggest influence on the prediction. This helps you understand *what drives results* in your data.
        """)

    train_tab, uplift_tab, model_info_tab = st.tabs(["Train Predictive Model", "Train Uplift Model", "Model Info"])

    with train_tab:
        st.markdown("""
        <div class="help-box">
            <p><strong>What does this do?</strong> Upload your data and the system will learn patterns from it. 
            Once trained, it can predict outcomes for new people — like whether a future user will convert, or how much they'll spend.
            You pick what to predict, and it figures out the best approach automatically.</p>
        </div>
        """, unsafe_allow_html=True)

        ml_file = st.file_uploader("Upload training data", type=['csv', 'xlsx', 'xls'], key="ml_train")

        if ml_file:
            try:
                if ml_file.name.endswith('.csv'):
                    df = pd.read_csv(ml_file)
                else:
                    df = pd.read_excel(ml_file)

                st.session_state.current_df = df

                col1, col2 = st.columns(2)
                with col1:
                    render_metric_card("Dataset Size", f"{df.shape[0]:,} rows x {df.shape[1]} cols", "")
                with col2:
                    numeric = df.select_dtypes(include=[np.number]).shape[1]
                    render_metric_card("Feature Types", f"{numeric} numeric, {df.shape[1]-numeric} categorical", "")

                with st.expander("Data Preview"):
                    st.dataframe(df.head(10), width='stretch')

                engine = UniversalMLEngine()
                detection = engine.auto_detect_columns(df)

                st.markdown("---")
                st.subheader("Model Configuration")
                st.caption("Tell the system what to predict. It will figure out the best approach automatically.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    default_target = detection['potential_target_cols'][0] if detection['potential_target_cols'] else df.columns[-1]
                    default_idx = list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1
                    target_col = st.selectbox("What to predict", df.columns, index=default_idx, key="pred_target",
                                            help="The column you want the model to learn to predict. This is your outcome — like 'converted', 'revenue', 'clicked', etc.")
                with col2:
                    model_type = st.selectbox("Model Type", ["auto", "classification", "regression"],
                                            help="Leave on 'auto' and the system decides. Classification = predicting categories (yes/no). Regression = predicting numbers (revenue, time).")
                with col3:
                    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05,
                                         help="How much data to hold back for testing. 0.2 means 20% of your data tests how good the model is, 80% trains it.")

                exclude_cols = st.multiselect("Exclude Columns (optional)", df.columns,
                                             help="Columns to leave out of the model — like user IDs, names, timestamps, or anything that shouldn't influence predictions.")

                default_name = ml_file.name.rsplit('.', 1)[0] + "_model"
                model_name = st.text_input("Model Name", value=default_name, help="Give your model a name so you can identify it later in the Predictions tab.")

                if st.button("Train Model", type="primary", width='stretch'):
                    with st.spinner("Training ML models... This may take a moment."):
                        engine = UniversalMLEngine()
                        df_train = df.drop(columns=exclude_cols, errors='ignore') if exclude_cols else df.copy()

                        results = engine.train_model(df_train, target_col, model_type=model_type, test_size=test_size)

                        st.session_state.ml_engine = engine
                        st.session_state.trained_model_info = results

                        st.success(f"Model trained successfully! Best: {results['best_model']} (Score: {results['best_score']:.4f})")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            render_metric_card("Best Model", results['best_model'].replace('_', ' ').title(), "", "success-card")
                        with col2:
                            score_label = "AUC-ROC" if results['task_type'] == 'classification' else "R-squared"
                            render_metric_card(score_label, f"{results['best_score']:.4f}", "best model performance", "info-card")
                        with col3:
                            render_metric_card("Features Used", str(results['n_features']), f"{results['n_train']} train / {results['n_test']} test")

                        st.markdown("---")
                        st.subheader("Model Comparison")
                        for name, res in results['model_results'].items():
                            with st.expander(f"{name.replace('_', ' ').title()}", expanded=True):
                                metrics_cols = st.columns(len(res) - 1)
                                for j, (key, val) in enumerate([(k, v) for k, v in res.items() if k != 'score']):
                                    with metrics_cols[j]:
                                        st.metric(key.replace('_', ' ').title(), f"{val:.4f}")

                        if engine.feature_importance:
                            st.subheader("Feature Importance")
                            best_fi = engine.feature_importance.get(results['best_model'], {})
                            if best_fi:
                                top_n = min(15, len(best_fi))
                                top_features = list(best_fi.keys())[:top_n]
                                top_values = list(best_fi.values())[:top_n]

                                fig = go.Figure(go.Bar(
                                    x=top_values[::-1],
                                    y=top_features[::-1],
                                    orientation='h',
                                    marker=dict(color='#3b82f6', line=dict(width=0))
                                ))
                                fig.update_layout(
                                    title="Top Features",
                                    template="plotly_dark",
                                    height=max(300, top_n * 30),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(15,23,42,0.5)',
                                    margin=dict(l=150, r=30, t=50, b=30),
                                    xaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
                                    yaxis=dict(gridcolor='rgba(148,163,184,0.1)')
                                )
                                st.plotly_chart(fig, width='stretch')

                        save_filename = sanitize_filename(model_name) + '.pkl'
                        engine.save(save_filename)
                        model_entry = {
                            'name': model_name.strip() or 'Untitled Model',
                            'filename': save_filename,
                            'type': 'Predictive',
                            'task': results.get('task_type', 'N/A'),
                            'best_model': results.get('best_model', 'N/A'),
                            'score': results.get('best_score', 0),
                            'features': results.get('n_features', 0),
                            'trained_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                        }
                        st.session_state.saved_models = [m for m in st.session_state.saved_models if m['filename'] != save_filename]
                        st.session_state.saved_models.append(model_entry)
                        st.session_state.active_model_name = model_entry['filename']
                        st.info(f"Model **{model_entry['name']}** saved. You can now use it in the Predictions tab.")

                        score_quality = "excellent" if results['best_score'] > 0.85 else ("good" if results['best_score'] > 0.7 else "moderate")
                        st.markdown(f"""
                        <div class="help-box" style="margin-top: 8px;">
                            <p><strong>What to do next:</strong> Your model has {score_quality} predictive power.
                            Go to the <strong>Predictions</strong> page to upload new data and score it with this model.
                            Check the Feature Importance chart above to see which factors matter most — this often reveals surprising insights about your users.</p>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with uplift_tab:
        st.markdown("""
        <div class="help-box">
            <p><strong>What does this do?</strong> An uplift model goes beyond just predicting outcomes — it figures out 
            <strong>which specific people benefit most from your treatment</strong>. Instead of treating everyone the same, you can 
            target the users who will respond best and skip those who won't be affected. This is how companies like Netflix and Amazon personalize their approach.</p>
        </div>
        """, unsafe_allow_html=True)

        uplift_file = st.file_uploader("Upload data with treatment/control groups", type=['csv', 'xlsx', 'xls'], key="uplift_train")

        if uplift_file:
            try:
                if uplift_file.name.endswith('.csv'):
                    df = pd.read_csv(uplift_file)
                else:
                    df = pd.read_excel(uplift_file)

                st.session_state.current_df = df

                with st.expander("Data Preview"):
                    st.dataframe(df.head(10), width='stretch')

                engine = UniversalMLEngine()
                detection = engine.auto_detect_columns(df)

                col1, col2 = st.columns(2)
                with col1:
                    default_group = detection['potential_group_cols'][0] if detection['potential_group_cols'] else df.columns[0]
                    default_idx = list(df.columns).index(default_group) if default_group in df.columns else 0
                    treatment_col = st.selectbox("Treatment Column", df.columns, index=default_idx, key="uplift_treat",
                                               help="The column that shows which group each person was in — e.g., 'control' vs 'treatment', 'A' vs 'B'.")
                with col2:
                    default_target = detection['potential_target_cols'][0] if detection['potential_target_cols'] else df.columns[-1]
                    default_idx = list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1
                    target_col = st.selectbox("Outcome Column", df.columns, index=default_idx, key="uplift_target",
                                            help="The result you measured — did they convert, how much did they spend, etc.")

                uplift_default_name = uplift_file.name.rsplit('.', 1)[0] + "_uplift_model"
                uplift_model_name = st.text_input("Model Name", value=uplift_default_name, key="uplift_model_name", help="Give your model a name so you can identify it later in the Predictions tab.")

                if st.button("Train Uplift Model", type="primary", width='stretch'):
                    with st.spinner("Training uplift model..."):
                        engine = UniversalMLEngine()
                        results = engine.train_uplift_model(df, target_col, treatment_col)

                        st.session_state.ml_engine = engine
                        st.session_state.trained_model_info = results

                        st.success("Uplift model trained!")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            render_metric_card("Avg Uplift", f"{results['avg_uplift']:.4f}", "mean treatment effect", "success-card" if results['avg_uplift'] > 0 else "danger-card")
                        with col2:
                            render_metric_card("Positive Uplift", f"{results['positive_uplift_pct']:.1f}%", "of population benefits")
                        with col3:
                            render_metric_card("Control Score", f"{results['score_control']:.4f}", f"{results['n_control']:,} samples")
                        with col4:
                            render_metric_card("Treatment Score", f"{results['score_treatment']:.4f}", f"{results['n_treatment']:,} samples")

                        uplift_scores = engine.predict_uplift(df)

                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=uplift_scores, nbinsx=50,
                            marker=dict(color='#3b82f6', line=dict(width=1, color='#1e293b'))
                        ))
                        fig.add_vline(x=0, line_dash="dash", line_color="#f59e0b", annotation_text="Zero Uplift")
                        fig.update_layout(
                            title="Uplift Score Distribution",
                            xaxis_title="Uplift Score",
                            yaxis_title="Count",
                            template="plotly_dark",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(15,23,42,0.5)',
                            margin=dict(t=50, b=50, l=50, r=30),
                        )
                        st.plotly_chart(fig, width='stretch')

                        if engine.feature_importance.get('uplift'):
                            st.subheader("Uplift Feature Importance")
                            fi = engine.feature_importance['uplift']
                            top_n = min(15, len(fi))
                            top_features = list(fi.keys())[:top_n]
                            top_values = list(fi.values())[:top_n]

                            fig = go.Figure(go.Bar(
                                x=top_values[::-1], y=top_features[::-1],
                                orientation='h',
                                marker=dict(color='#8b5cf6', line=dict(width=0))
                            ))
                            fig.update_layout(
                                title="Top Features Driving Uplift",
                                template="plotly_dark",
                                height=max(300, top_n * 30),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(15,23,42,0.5)',
                                margin=dict(l=150, r=30, t=50, b=30),
                            )
                            st.plotly_chart(fig, width='stretch')

                        uplift_save_filename = sanitize_filename(uplift_model_name) + '.pkl'
                        engine.save(uplift_save_filename)
                        model_entry = {
                            'name': uplift_model_name.strip() or 'Untitled Uplift Model',
                            'filename': uplift_save_filename,
                            'type': 'Uplift',
                            'task': 'uplift',
                            'best_model': 'Uplift (T-Learner)',
                            'score': results.get('avg_uplift', 0),
                            'features': len(engine.feature_names) if engine.feature_names else 0,
                            'trained_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                        }
                        st.session_state.saved_models = [m for m in st.session_state.saved_models if m['filename'] != uplift_save_filename]
                        st.session_state.saved_models.append(model_entry)
                        st.session_state.active_model_name = model_entry['filename']
                        st.info(f"Model **{model_entry['name']}** saved. Use the Predictions tab to score new data.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with model_info_tab:
        if st.session_state.trained_model_info:
            info = st.session_state.trained_model_info
            active_entry_info = next((m for m in st.session_state.saved_models if m['filename'] == st.session_state.active_model_name), None)
            display_name = active_entry_info['name'] if active_entry_info else "Unnamed Model"
            st.subheader(f"Model: {display_name}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**General Info**")
                st.json({
                    'model_name': display_name,
                    'task_type': info.get('task_type', 'N/A'),
                    'best_model': info.get('best_model', info.get('learner_type', 'N/A')),
                    'n_features': info.get('n_features', 0),
                })
            with col2:
                st.markdown("**Performance**")
                if 'model_results' in info:
                    for name, res in info['model_results'].items():
                        st.write(f"**{name}**: score = {res.get('score', 0):.4f}")
                elif 'avg_uplift' in info:
                    st.write(f"Avg Uplift: {info['avg_uplift']:.4f}")
                    st.write(f"Positive Uplift %: {info.get('positive_uplift_pct', 0):.1f}%")

            if 'feature_names' in info:
                with st.expander("Feature Names"):
                    st.write(info['feature_names'])
        else:
            st.info("No model trained yet. Go to 'Train Predictive Model' or 'Train Uplift Model' to get started.")


elif page == "Predictions":
    st.markdown('<div class="section-header">Make Predictions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="help-box">
        <p><strong>What does this do?</strong> After training a model in the ML Studio, you can use it here to score new data.
        Upload a file with the same kind of columns as your training data, and the model will predict the outcome for each row 
        — like whether each person will convert, how much they'll spend, or who to target with your campaign. No need to retrain.</p>
    </div>
    """, unsafe_allow_html=True)

    saved_models = st.session_state.saved_models

    if saved_models:
        st.markdown("#### Select a Model")
        model_display = [f"{m['name']} ({m['type']}, trained {m['trained_at']})" for m in reversed(saved_models)]
        models_reversed = list(reversed(saved_models))
        selected_idx = st.selectbox("Saved Models", range(len(model_display)), index=0, format_func=lambda i: model_display[i], help="Pick which model to use for predictions. The most recently trained model is shown first.")
        selected_model = models_reversed[selected_idx]

        if st.session_state.active_model_name != selected_model['filename'] or st.session_state.ml_engine is None or not st.session_state.ml_engine.is_trained:
            if os.path.exists(selected_model['filename']):
                engine = UniversalMLEngine()
                engine.load(selected_model['filename'])
                st.session_state.ml_engine = engine
                st.session_state.active_model_name = selected_model['filename']
            else:
                st.error(f"Model file '{selected_model['filename']}' not found. It may have been deleted. Please retrain.")
                engine = None
        else:
            engine = st.session_state.ml_engine
    else:
        engine = st.session_state.ml_engine
        if engine is None and os.path.exists('trained_model.pkl'):
            engine = UniversalMLEngine()
            engine.load('trained_model.pkl')
            st.session_state.ml_engine = engine

    if engine and engine.is_trained:
        active_entry = next((m for m in saved_models if m['filename'] == st.session_state.active_model_name), None)
        active_display_name = active_entry['name'] if active_entry else "Unnamed Model"

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Active Model", active_display_name, f"Task: {engine.task_type}", "success-card")
        with col2:
            has_uplift = 'uplift_control' in engine.models
            model_type = "Uplift Model" if has_uplift else "Predictive Model"
            render_metric_card("Model Type", model_type, f"{len(engine.feature_names)} features")
        with col3:
            trained_at = active_entry['trained_at'] if active_entry else "Unknown"
            render_metric_card("Trained", trained_at, active_entry.get('best_model', '') if active_entry else '', "info-card")

        st.markdown("---")

        pred_file = st.file_uploader("Upload data for prediction", type=['csv', 'xlsx', 'xls'], key="pred_file")

        if pred_file:
            try:
                if pred_file.name.endswith('.csv'):
                    pred_df = pd.read_csv(pred_file)
                else:
                    pred_df = pd.read_excel(pred_file)

                st.write(f"Loaded {pred_df.shape[0]:,} rows x {pred_df.shape[1]} columns")

                with st.expander("Data Preview"):
                    st.dataframe(pred_df.head(10), width='stretch')

                col1, col2 = st.columns(2)
                with col1:
                    pred_type = st.radio("Prediction Type",
                                        ["Standard Prediction", "Uplift Prediction"] if has_uplift else ["Standard Prediction"])
                with col2:
                    if pred_type == "Standard Prediction":
                        model_choice = st.selectbox("Model", [k for k in engine.models.keys() if k not in ['uplift_control', 'uplift_treatment']])
                    else:
                        model_choice = None

                if st.button("Generate Predictions", type="primary", width='stretch'):
                    with st.spinner("Generating predictions..."):
                        try:
                            if pred_type == "Uplift Prediction":
                                scores = engine.predict_uplift(pred_df)
                                pred_df['uplift_score'] = scores
                                pred_df['recommended_action'] = np.where(scores > 0, 'Target', 'Skip')

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    render_metric_card("Avg Uplift", f"{np.mean(scores):.4f}", "", "info-card")
                                with col2:
                                    render_metric_card("Target Rate", f"{np.mean(scores > 0)*100:.1f}%", "recommended for treatment")
                                with col3:
                                    render_metric_card("Max Uplift", f"{np.max(scores):.4f}", "best responder")

                            else:
                                scores = engine.predict(pred_df, model_choice)
                                if engine.task_type == 'classification':
                                    pred_df['prediction_probability'] = scores
                                    pred_df['predicted_class'] = (scores >= 0.5).astype(int)
                                else:
                                    pred_df['predicted_value'] = scores

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    render_metric_card("Mean Prediction", f"{np.mean(scores):.4f}", "")
                                with col2:
                                    render_metric_card("Std Dev", f"{np.std(scores):.4f}", "prediction spread")
                                with col3:
                                    render_metric_card("Predictions", f"{len(scores):,}", "rows scored")

                            st.markdown("---")
                            st.subheader("Results")
                            st.dataframe(pred_df, width='stretch', height=400)

                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=scores, nbinsx=50,
                                marker=dict(color='#3b82f6', line=dict(width=1, color='#1e293b'))
                            ))
                            fig.update_layout(
                                title="Prediction Distribution",
                                xaxis_title="Score",
                                yaxis_title="Count",
                                template="plotly_dark",
                                height=350,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(15,23,42,0.5)',
                            )
                            st.plotly_chart(fig, width='stretch')

                            csv_out = pred_df.to_csv(index=False)
                            st.download_button("Download Predictions", csv_out, "predictions.csv", "text/csv",
                                              width='stretch')

                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    else:
        st.markdown("""
        <div class="metric-card warning-card" style="text-align: center; padding: 40px;">
            <h3 style="text-transform: none !important; font-size: 18px !important; color: #f59e0b !important;">No model trained yet</h3>
            <p class="sub" style="font-size: 15px !important; line-height: 1.7;">
                To make predictions, you first need to train a model in the <strong style="color: #f1f5f9;">ML Model Studio</strong>. 
                Upload your experiment data there, pick what to predict, and hit Train. Once done, come back here and your model will be ready to use.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Go to ML Model Studio"):
            st.session_state['_nav_target'] = 'ML Model Studio'
            st.rerun()


elif page == "Sample Datasets":
    st.markdown('<div class="section-header">Sample Datasets</div>', unsafe_allow_html=True)
    st.markdown("Don't have your own data yet? No problem! Try the platform with these ready-made example datasets from different industries.")

    samples = SampleDataGenerator.get_all_samples()

    cols = st.columns(3)
    for idx, (key, sample) in enumerate(samples.items()):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"### {sample['name']}")
                st.caption(sample['description'])
                st.caption(f"{sample['df'].shape[0]:,} rows | {sample['df'].shape[1]} columns")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Analyze", key=f"analyze_{key}", width='stretch'):
                        with st.spinner("Analyzing..."):
                            group_col = sample.get("group_col", "group")
                            results = run_ab_analysis(
                                sample['df'], group_col,
                                sample['metric_col'], "auto", key
                            )
                            if results:
                                st.session_state.predictions.append(results)
                                st.success(f"Done! Check AB Test Analysis > Results.")
                with c2:
                    csv_data = sample['df'].to_csv(index=False)
                    st.download_button("Download", csv_data, f"sample_{key}.csv", "text/csv",
                                      key=f"dl_{key}", width='stretch')

    st.markdown("---")
    st.subheader("Generate Custom Training Data")

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.number_input("Number of Samples", 1000, 500000, 50000, 10000)
    with col2:
        gen_domain = st.selectbox("Domain", ["All Domains", "tech", "ecommerce", "marketing", "gaming", "finance", "healthcare"])

    if st.button("Generate Training Dataset", type="primary", width='stretch'):
        with st.spinner(f"Generating {n_samples:,} samples..."):
            from enhanced_data_generator import EnhancedDataGenerator
            if gen_domain == "All Domains":
                gen_df = EnhancedDataGenerator.generate_multi_domain_training_data(n_samples)
            else:
                gen_df = EnhancedDataGenerator._generate_domain_data(gen_domain, n_samples)

            st.session_state.current_df = gen_df
            st.success(f"Generated {gen_df.shape[0]:,} rows x {gen_df.shape[1]} columns")

            with st.expander("Preview", expanded=True):
                st.dataframe(gen_df.head(20), width='stretch')

            csv_gen = gen_df.to_csv(index=False)
            st.download_button("Download Generated Data", csv_gen, f"generated_{gen_domain}_{n_samples}.csv",
                              "text/csv", width='stretch')


st.divider()
st.markdown("""
<div style='text-align: center; color: #475569; font-size: 12px; padding: 10px;'>
    AB Testing Pro v2.0 | Built by Harsha | ML-Powered Analysis Platform
</div>
""", unsafe_allow_html=True)
