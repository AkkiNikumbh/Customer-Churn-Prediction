"""
app.py
Streamlit Dashboard for Customer Churn & Retention Analysis.
Uses 55,000 data points sample matching the 85%+ accuracy resume benchmark.
Provides Executive Summary, Exploratory Analysis, Predictive Simulator, and At-Risk Export.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn & Retention Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM INJECTED STYLING ---
st.markdown("""
    <style>
    /* Global Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #F8FAFC;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #CBD5E1;
        margin-top: 4px;
    }
    /* Risk Badges */
    .risk-badge-high {
        background-color: #EF4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .risk-badge-med {
        background-color: #F59E0B;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)


# --- DATA & MODEL INITIALIZATION ---
@st.cache_data
def load_dataset(sample_size=55000):
    train_file = "customer_churn_dataset-training-master.csv"
    if os.path.exists(train_file):
        full_df = pd.read_csv(train_file).dropna()
        if len(full_df) > sample_size:
            df = full_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            df = full_df.copy()
    else:
        st.error(f"Dataset file '{train_file}' not found in workspace!")
        st.stop()
    
    numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)
    df["Churn_Label"] = df["Churn"].apply(lambda x: "Yes" if x == 1 else "No")
    return df

def load_or_create_model():
    model_file = "churn_model.joblib"
    metrics_file = "model_metrics.json"
    if not os.path.exists(model_file) or not os.path.exists(metrics_file):
        from model_pipeline import run_model_pipeline
        pipeline, metrics = run_model_pipeline()
    else:
        pipeline = joblib.load(model_file)
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    return pipeline, metrics

df_raw = load_dataset(sample_size=55000)
pipeline, metrics = load_or_create_model()


# --- SIDEBAR FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=65)
st.sidebar.title("Filter Options")
st.sidebar.markdown(f"**Sample Dataset:** {len(df_raw):,} Customers")

contract_options = ["All"] + sorted(list(df_raw["Contract Length"].dropna().unique()))
selected_contract = st.sidebar.selectbox("Contract Length", contract_options)

sub_options = ["All"] + sorted(list(df_raw["Subscription Type"].dropna().unique()))
selected_sub = st.sidebar.selectbox("Subscription Type", sub_options)

gender_options = ["All"] + sorted(list(df_raw["Gender"].dropna().unique()))
selected_gender = st.sidebar.selectbox("Gender", gender_options)

min_tenure = int(df_raw["Tenure"].min())
max_tenure = int(df_raw["Tenure"].max())
selected_tenure_range = st.sidebar.slider(
    "Tenure Range (Months)", 
    min_value=min_tenure, 
    max_value=max_tenure, 
    value=(min_tenure, max_tenure)
)

# Apply Sidebar Filters
df_filtered = df_raw.copy()
if selected_contract != "All":
    df_filtered = df_filtered[df_filtered["Contract Length"] == selected_contract]

if selected_sub != "All":
    df_filtered = df_filtered[df_filtered["Subscription Type"] == selected_sub]

if selected_gender != "All":
    df_filtered = df_filtered[df_filtered["Gender"] == selected_gender]

df_filtered = df_filtered[
    (df_filtered["Tenure"] >= selected_tenure_range[0]) & 
    (df_filtered["Tenure"] <= selected_tenure_range[1])
]

# --- MAIN DASHBOARD HEADER ---
st.title("📊 Customer Churn & Retention Analytics")
st.caption(f"ML-powered customer churn insights analyzed on {len(df_raw):,} dataset sample matching **{metrics['accuracy']*100:.1f}% Model Accuracy**.")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Summary", 
    "🔍 Exploratory Analysis", 
    "🔮 Predictive Simulator", 
    "🎯 At-Risk Export"
])


# ==========================================
# PAGE SECTION 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    st.subheader("Key Performance Indicators (KPIs)")
    
    total_customers = len(df_filtered)
    churned_customers = len(df_filtered[df_filtered["Churn"] == 1])
    churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
    
    total_spend_lost = df_filtered[df_filtered["Churn"] == 1]["Total Spend"].sum()
    high_risk_count = len(df_filtered[(df_filtered["Churn"] == 1) | (df_filtered["Support Calls"] >= 5) | (df_filtered["Payment Delay"] >= 15)])
    
    projected_savings = total_spend_lost * 0.12
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Overall Churn Rate</div>
                <div class="metric-value" style="color: {'#EF4444' if churn_rate > 30 else '#38BDF8'};">{churn_rate:.1f}%</div>
                <div class="metric-subtext">{churned_customers:,} / {total_customers:,} Customers</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Revenue Lost</div>
                <div class="metric-value" style="color: #F59E0B;">${total_spend_lost:,.0f}</div>
                <div class="metric-subtext">Sum of Total Spend (Churned)</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Projected Revenue Saved</div>
                <div class="metric-value" style="color: #10B981;">${projected_savings:,.0f}</div>
                <div class="metric-subtext">12% Annual Churn Reduction</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">High-Risk Accounts</div>
                <div class="metric-value" style="color: #EC4899;">{high_risk_count:,}</div>
                <div class="metric-subtext">Require Active Outreach</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Model Accuracy</div>
                <div class="metric-value" style="color: #38BDF8;">{metrics['accuracy']*100:.1f}%</div>
                <div class="metric-subtext">Precision: {metrics['precision']*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Charts Row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("##### Churn Distribution by Contract Length")
        contract_churn = df_filtered.groupby(["Contract Length", "Churn_Label"]).size().reset_index(name="Count")
        fig_contract = px.bar(
            contract_churn,
            x="Contract Length",
            y="Count",
            color="Churn_Label",
            barmode="group",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            template="plotly_dark",
            height=340
        )
        fig_contract.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_contract, use_container_width=True)

    with col_right:
        st.markdown("##### Churn Breakdown by Subscription Type")
        sub_churn = df_filtered.groupby(["Subscription Type", "Churn_Label"]).size().reset_index(name="Count")
        fig_sub = px.bar(
            sub_churn,
            x="Subscription Type",
            y="Count",
            color="Churn_Label",
            barmode="group",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            template="plotly_dark",
            height=340
        )
        fig_sub.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_sub, use_container_width=True)


# ==========================================
# PAGE SECTION 2: EXPLORATORY ANALYSIS
# ==========================================
with tab2:
    st.subheader("Exploratory Data Analysis & Key Churn Drivers")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("##### 1. Tenure Distribution vs. Churn")
        fig_tenure = px.histogram(
            df_filtered,
            x="Tenure",
            color="Churn_Label",
            nbins=30,
            marginal="box",
            opacity=0.75,
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            template="plotly_dark",
            height=380,
            labels={"Tenure": "Tenure (Months)", "count": "Customer Count"}
        )
        fig_tenure.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_tenure, use_container_width=True)
        
    with eda_col2:
        st.markdown("##### 2. Total Spend vs. Churn")
        fig_spend = px.box(
            df_filtered,
            x="Churn_Label",
            y="Total Spend",
            color="Churn_Label",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            template="plotly_dark",
            height=380,
            labels={"Total Spend": "Total Spend ($)", "Churn_Label": "Churn Status"}
        )
        fig_spend.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_spend, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    eda_col3, eda_col4 = st.columns(2)
    
    with eda_col3:
        st.markdown("##### 3. Support Calls Volume vs. Churn Rate")
        call_stats = df_filtered.groupby("Support Calls")["Churn"].apply(
            lambda x: (x == 1).mean() * 100
        ).reset_index(name="ChurnRate")
        
        fig_calls = px.line(
            call_stats,
            x="Support Calls",
            y="ChurnRate",
            markers=True,
            line_shape="linear",
            template="plotly_dark",
            height=380,
            labels={"Support Calls": "Number of Support Calls", "ChurnRate": "Churn Rate (%)"}
        )
        fig_calls.update_traces(line_color="#F59E0B", line_width=3, marker=dict(size=8, color="#EF4444"))
        fig_calls.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_calls, use_container_width=True)

    with eda_col4:
        st.markdown("##### 4. Random Forest Feature Importances")
        feat_df = pd.DataFrame(metrics["top_features"]).head(10).sort_values("importance", ascending=True)
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_dark",
            height=380,
            labels={"importance": "Importance Weight", "feature": "Model Feature"}
        )
        fig_feat.update_traces(marker_color="#38BDF8")
        fig_feat.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_feat, use_container_width=True)


# ==========================================
# PAGE SECTION 3: PREDICTIVE SIMULATOR
# ==========================================
with tab3:
    st.subheader("Customer Churn Risk Predictor")
    st.markdown("Input customer profile attributes to assess real-time churn probability.")

    with st.form("churn_prediction_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            input_age = st.number_input("Age", min_value=18, max_value=100, value=35)
            input_gender = st.selectbox("Gender", ["Female", "Male"])
            input_tenure = st.number_input("Tenure (Months)", min_value=1, max_value=60, value=12)
            input_usage = st.number_input("Usage Frequency", min_value=1, max_value=30, value=15)

        with f_col2:
            input_calls = st.slider("Support Calls", min_value=0, max_value=10, value=3)
            input_delay = st.slider("Payment Delay (Days)", min_value=0, max_value=30, value=5)
            input_sub = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

        with f_col3:
            input_contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            input_spend = st.number_input("Total Spend ($)", min_value=100.0, max_value=1000.0, value=500.0)
            input_interaction = st.number_input("Last Interaction (Days ago)", min_value=1, max_value=30, value=10)

        submit_predict = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

    if submit_predict:
        # Construct single row dataframe matching feature pipeline schema
        input_data = pd.DataFrame([{
            "Age": input_age,
            "Gender": input_gender,
            "Tenure": input_tenure,
            "Usage Frequency": input_usage,
            "Support Calls": input_calls,
            "Payment Delay": input_delay,
            "Subscription Type": input_sub,
            "Contract Length": input_contract,
            "Total Spend": input_spend,
            "Last Interaction": input_interaction
        }])

        proba = pipeline.predict_proba(input_data)[0][1]
        risk_pct = proba * 100

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### Prediction Result")
            if risk_pct >= 60:
                badge_html = f'<div class="risk-badge-high">HIGH RISK ({risk_pct:.1f}%)</div>'
            elif risk_pct >= 35:
                badge_html = f'<div class="risk-badge-med">MEDIUM RISK ({risk_pct:.1f}%)</div>'
            else:
                badge_html = f'<div class="risk-badge-low">LOW RISK ({risk_pct:.1f}%)</div>'

            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge Chart
            fig_gauge = gg.Figure(gg.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={'suffix': "%", 'font': {'size': 36}},
                title={'text': "Churn Probability Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if risk_pct >= 60 else ("#F59E0B" if risk_pct >= 35 else "#10B981")},
                    'steps': [
                        {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [35, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### Recommended Retention Actions")
            if risk_pct >= 60:
                st.error("🚨 **Urgent Outreach Required!**")
                st.markdown("""
                - **Contract Upgrade:** Offer a 15% discount on converting to an Annual Contract.
                - **Dedicated Tech Support:** Assign priority customer success manager to resolve support calls.
                - **Payment Plan:** Address payment delays with flexible billing reminders or auto-pay incentives.
                """)
            elif risk_pct >= 35:
                st.warning("⚠️ **Proactive Engagement Recommended**")
                st.markdown("""
                - Send a satisfaction survey regarding recent support inquiries.
                - Offer automated payment setup incentives (e.g. $5 bill credit).
                - Highlight value of upgrade options or long-term contract discounts.
                """)
            else:
                st.success("✅ **Healthy Customer Account**")
                st.markdown("""
                - Candidate for upsell/cross-sell (e.g. Premium tier upgrade).
                - Maintain standard customer feedback loop.
                """)


# ==========================================
# PAGE SECTION 4: AT-RISK EXPORT
# ==========================================
with tab4:
    st.subheader("High-Risk Customer Outreach Queue")
    st.markdown("Filter and export high-risk accounts to pass directly to customer success and retention teams.")

    # Predict probabilities for table
    X_outreach = df_filtered.drop(columns=["CustomerID", "Churn", "Churn_Label"], errors="ignore")
    probas = pipeline.predict_proba(X_outreach)[:, 1]
    
    df_outreach = df_filtered.copy()
    df_outreach["Churn_Risk_%"] = np.round(probas * 100, 1)
    df_outreach["Risk_Category"] = pd.cut(
        df_outreach["Churn_Risk_%"], 
        bins=[-1, 35, 60, 100], 
        labels=["Low", "Medium", "High"]
    )

    filter_risk_level = st.multiselect(
        "Filter Risk Levels", 
        ["High", "Medium", "Low"], 
        default=["High", "Medium"]
    )

    df_export = df_outreach[df_outreach["Risk_Category"].isin(filter_risk_level)].sort_values(
        by="Churn_Risk_%", ascending=False
    )

    st.markdown(f"**Found {len(df_export):,} customers matching risk filter criteria.**")

    cols_to_display = [c for c in [
        "CustomerID", "Churn_Risk_%", "Risk_Category", "Contract Length", 
        "Subscription Type", "Tenure", "Total Spend", "Support Calls", "Payment Delay"
    ] if c in df_export.columns]

    st.dataframe(
        df_export[cols_to_display],
        use_container_width=True,
        hide_index=True
    )

    # Download CSV Button
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV for Retention Outreach",
        data=csv_data,
        file_name="high_risk_retention_list.csv",
        mime="text/csv",
        use_container_width=True
    )
