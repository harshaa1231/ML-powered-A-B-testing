# AB Testing Pro v2.0
# Author: Harsha
# ML-Powered A/B Testing & Analysis Platform

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
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
        padding: 24px;
        border: 1px solid rgba(59, 130, 246, 0.15);
        text-align: center;
        height: 100%;
    }
    .feature-card .icon {
        font-size: 36px;
        margin-bottom: 12px;
    }
    .feature-card h4 {
        color: #e2e8f0 !important;
        margin: 0 0 8px 0;
    }
    .feature-card p {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
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

    page = st.radio(
        "Navigation",
        ["Home", "Simple A/B Test", "AB Test Analysis", "ML Model Studio", "Predictions", "Sample Datasets"],
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
    <div class="hero-section">
        <h1>AB Testing Pro</h1>
        <p>Find out what works and what doesn't — powered by data, explained in plain English.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="border-left-color: #60a5fa;">
        <h3 style="text-transform: none !important; font-size: 16px !important;">What is A/B Testing?</h3>
        <p class="sub" style="font-size: 15px !important; line-height: 1.8;">
            Imagine you have two versions of something — maybe two different website designs, two email subject lines,
            or two pricing options. You show Version A to one group of people and Version B to another. Then you compare
            the results to see which one performed better.
            <br><br>
            <strong style="color: #f1f5f9;">That's it. That's A/B testing.</strong>
            <br><br>
            This tool helps you go beyond guessing. It uses math and machine learning to tell you whether the difference
            you're seeing is real or just random luck — and explains everything in language anyone can understand.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("### Where to start")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="feature-card">
            <div class="icon">✨</div>
            <h4>New to A/B testing?</h4>
            <p>Go to <strong>Simple A/B Test</strong> in the sidebar. Just type in your numbers and get a plain-English answer. No files or technical knowledge needed.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="feature-card">
            <div class="icon">📊</div>
            <h4>Have a data file?</h4>
            <p>Go to <strong>AB Test Analysis</strong> to upload your CSV or Excel file. The tool auto-detects your columns and picks the right test for you.</p>
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""<div class="feature-card">
            <div class="icon">🤖</div>
            <h4>Want deeper insights?</h4>
            <p>Use the <strong>ML Model Studio</strong> to train a machine learning model. It learns patterns in your data and predicts future outcomes.</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="feature-card">
            <div class="icon">🧪</div>
            <h4>Just exploring?</h4>
            <p>Check out <strong>Sample Datasets</strong> to play with pre-built examples from tech, ecommerce, marketing, and more.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    if st.session_state.predictions:
        st.markdown('<div class="section-header">Recent Results</div>', unsafe_allow_html=True)

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
            render_metric_card("Avg P-Value", f"{avg_p:.4f}", "lower = more significant", "info-card")

    else:
        st.info("Get started by navigating to **Simple A/B Test** for a quick comparison, or try **Sample Datasets** for a demo.")


elif page == "Simple A/B Test":
    st.markdown('<div class="section-header">Simple A/B Test</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="border-left-color: #60a5fa;">
        <h3 style="text-transform: none !important; font-size: 16px !important;">How does this work?</h3>
        <p class="sub" style="font-size: 15px !important; line-height: 1.6;">
            You tried two different versions of something (Version A and Version B). Now you want to know:
            <strong style="color: #f1f5f9;">did one actually perform better, or was it just luck?</strong>
            Enter your numbers below and we'll tell you in plain English.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    simple_mode = st.radio(
        "What did you test?",
        ["Something people click or sign up for (yes/no outcome)", "Something you can measure with numbers (like revenue or time spent)"],
        help="Pick the one that best describes your experiment"
    )

    st.markdown("---")

    if "yes/no" in simple_mode:
        st.markdown("#### Enter your results")
        st.caption("For example: You showed two versions of a button. How many people saw each version, and how many clicked?")

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
    with st.expander("New to A/B testing? Here's a quick guide"):
        st.markdown("""
**What is A/B testing?**
It's like a science experiment for your business. You show one group of people Version A (the original) and another group Version B (the new version), then compare which one does better.

**What's a "conversion"?**
Any action you want people to take — clicking a button, making a purchase, signing up, opening an email, etc.

**What does "statistically significant" mean?**
It means we're confident the difference is real and not just random luck. Think of it like flipping a coin — if you flip 10 times and get 6 heads, that could be luck. But if you flip 1000 times and get 600 heads, something is definitely going on.

**How many people do I need?**
More is better. As a rough guide:
- Small differences need more people to detect (thousands)
- Big differences can be spotted with fewer people (hundreds)
- A minimum of 100 per group is a good starting point

**When should I use the Advanced AB Test Analysis instead?**
Use the advanced page when you have a full dataset (CSV or Excel file) with individual-level data, not just summary numbers. It can auto-detect your columns and run more sophisticated tests.
        """)


elif page == "AB Test Analysis":
    st.markdown('<div class="section-header">AB Test Analysis (Advanced)</div>', unsafe_allow_html=True)
    st.caption("Upload a dataset file and run detailed statistical tests. Best for when you have raw experiment data in a CSV or Excel file.")

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
                st.subheader("Configure Test")

                engine = UniversalMLEngine()
                detection = engine.auto_detect_columns(df)

                col1, col2 = st.columns(2)
                with col1:
                    default_group = detection['potential_group_cols'][0] if detection['potential_group_cols'] else df.columns[0]
                    default_idx = list(df.columns).index(default_group) if default_group in df.columns else 0
                    group_col = st.selectbox("Group Column (control vs treatment)", df.columns, index=default_idx,
                                            help="Column that identifies which group each row belongs to")
                with col2:
                    default_target = detection['potential_target_cols'][0] if detection['potential_target_cols'] else df.columns[-1]
                    default_idx = list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1
                    metric_col = st.selectbox("Metric Column (outcome to measure)", df.columns, index=default_idx,
                                             help="The outcome metric you want to compare between groups")

                col1, col2 = st.columns(2)
                with col1:
                    test_type = st.selectbox("Test Type", ["auto", "ttest", "chi_square", "mann_whitney"],
                                           help="'auto' will pick the best test for your data")
                with col2:
                    domain = st.selectbox("Domain", ["general", "tech", "ecommerce", "marketing", "gaming", "finance", "healthcare"])

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
            for i, pred in enumerate(reversed(st.session_state.predictions)):
                with st.expander(f"Test {len(st.session_state.predictions) - i}: {pred.get('metric', 'N/A')} ({pred.get('domain', '')})", expanded=(i == 0)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        p_val = pred.get('p_value', 1)
                        sig_class = "success-card" if p_val < 0.05 else "danger-card"
                        render_metric_card("P-Value", f"{p_val:.6f}", "< 0.05 = significant", sig_class)
                    with col2:
                        render_metric_card("Effect Size", f"{pred.get('effect_size', 0):.4f}", pred.get('test_name', ''))
                    with col3:
                        up = pred.get('uplift_percentage', 0)
                        up_class = "success-card" if up > 0 else "danger-card"
                        render_metric_card("Uplift", f"{up:.2f}%", "treatment vs control", up_class)
                    with col4:
                        sig = pred.get('is_significant', False)
                        badge = '<span class="result-badge badge-success">SIGNIFICANT</span>' if sig else '<span class="result-badge badge-danger">NOT SIGNIFICANT</span>'
                        st.markdown(f"<div style='padding-top: 30px; text-align: center;'>{badge}</div>", unsafe_allow_html=True)

                    if 'n_control' in pred:
                        st.caption(f"Control: {pred['n_control']:,} samples | Treatment: {pred['n_treatment']:,} samples")

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

    train_tab, uplift_tab, model_info_tab = st.tabs(["Train Predictive Model", "Train Uplift Model", "Model Info"])

    with train_tab:
        st.markdown("Upload any dataset and train a model that can **predict outcomes for new data**. The system auto-detects your column types and picks the best algorithm.")

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

                col1, col2, col3 = st.columns(3)
                with col1:
                    default_target = detection['potential_target_cols'][0] if detection['potential_target_cols'] else df.columns[-1]
                    default_idx = list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1
                    target_col = st.selectbox("Target Column (what to predict)", df.columns, index=default_idx, key="pred_target")
                with col2:
                    model_type = st.selectbox("Model Type", ["auto", "classification", "regression"],
                                            help="'auto' detects based on target values")
                with col3:
                    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

                exclude_cols = st.multiselect("Exclude Columns (optional)", df.columns,
                                             help="Columns to exclude from features (e.g., IDs, dates)")

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

                        engine.save('trained_model.pkl')
                        st.info("Model saved. You can now use it in the Predictions tab.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with uplift_tab:
        st.markdown("Train an **uplift model** to find out **who benefits most** from your treatment. This tells you which users to target and which to leave alone.")

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
                    treatment_col = st.selectbox("Treatment Column", df.columns, index=default_idx, key="uplift_treat")
                with col2:
                    default_target = detection['potential_target_cols'][0] if detection['potential_target_cols'] else df.columns[-1]
                    default_idx = list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1
                    target_col = st.selectbox("Target Column", df.columns, index=default_idx, key="uplift_target")

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

                        engine.save('trained_model.pkl')
                        st.info("Uplift model saved. Use the Predictions tab to score new data.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with model_info_tab:
        if st.session_state.trained_model_info:
            info = st.session_state.trained_model_info
            st.subheader("Current Model Details")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**General Info**")
                st.json({
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
    st.caption("Use your trained model to score new data. Upload a file and get predictions for each row — no need to retrain.")

    engine = st.session_state.ml_engine

    if engine is None and os.path.exists('trained_model.pkl'):
        engine = UniversalMLEngine()
        engine.load('trained_model.pkl')
        st.session_state.ml_engine = engine

    if engine and engine.is_trained:
        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("Model Status", "Ready", f"Task: {engine.task_type}", "success-card")
        with col2:
            has_uplift = 'uplift_control' in engine.models
            model_type = "Uplift Model" if has_uplift else "Predictive Model"
            render_metric_card("Model Type", model_type, f"{len(engine.feature_names)} features")

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
        st.warning("No trained model available. Please train a model first in the ML Model Studio.")

        if st.button("Go to ML Model Studio"):
            st.session_state['page'] = 'ML Model Studio'
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
