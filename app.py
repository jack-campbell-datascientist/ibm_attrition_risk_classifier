import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px

st.set_page_config(page_title="Workforce Attrition Risk & Retention Simulator", layout="wide")

DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = "rf_attrition_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURES_PATH = "feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, encoders, feature_cols


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def encode_dataframe(raw_df, encoders, feature_cols):
    d = raw_df.copy()
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    d = d.drop(columns=[c for c in drop_cols if c in d.columns])
    if "Attrition" in d.columns:
        d = d.drop(columns=["Attrition"])
    for col, le in encoders.items():
        if col in d.columns:
            d[col] = le.transform(d[col].astype(str))
    return d[feature_cols]


model, encoders, feature_cols = load_artifacts()
df = load_data()

X_all = encode_dataframe(df, encoders, feature_cols)
df["PredictedRiskProb"] = model.predict_proba(X_all)[:, 1]

st.title("Workforce Attrition Risk & Retention Simulator")
st.caption(
    "Attrition risk modeling, department-level insight, and a scenario simulator "
    "for retention planning."
)

tab1, tab2, tab3 = st.tabs(["Executive Summary", "Department Drilldown", "Retention Simulator"])

# ---------------- TAB 1: EXECUTIVE SUMMARY ----------------
with tab1:
    total_headcount = len(df)
    historical_rate = (df["Attrition"] == "Yes").mean()
    at_risk_count = (df["PredictedRiskProb"] >= 0.5).sum()
    avg_risk = df["PredictedRiskProb"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Headcount", f"{total_headcount:,}")
    c2.metric("Historical Attrition Rate", f"{historical_rate:.1%}")
    c3.metric("Predicted At-Risk Employees", f"{at_risk_count:,}")
    c4.metric("Average Predicted Risk", f"{avg_risk:.1%}")

    st.subheader("Predicted Attrition Risk by Department")
    dept_risk = (
        df.groupby("Department")["PredictedRiskProb"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        dept_risk, x="Department", y="PredictedRiskProb",
        labels={"PredictedRiskProb": "Avg Predicted Risk"},
        color="PredictedRiskProb", color_continuous_scale="Reds"
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Drivers of Attrition Risk")
    importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(10)
    )
    fig2 = px.bar(
        importance[::-1], orientation="h",
        labels={"value": "Importance", "index": "Feature"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 2: DEPARTMENT DRILLDOWN ----------------
with tab2:
    st.subheader("Filter by Department and Role")
    dept_filter = st.selectbox("Department", ["All"] + sorted(df["Department"].unique().tolist()))
    filtered = df if dept_filter == "All" else df[df["Department"] == dept_filter]

    role_filter = st.selectbox("Job Role", ["All"] + sorted(filtered["JobRole"].unique().tolist()))
    if role_filter != "All":
        filtered = filtered[filtered["JobRole"] == role_filter]

    c1, c2, c3 = st.columns(3)
    c1.metric("Employees in View", f"{len(filtered):,}")
    c2.metric("Historical Attrition Rate", f"{(filtered['Attrition'] == 'Yes').mean():.1%}")
    c3.metric("Avg Predicted Risk", f"{filtered['PredictedRiskProb'].mean():.1%}")

    st.subheader("Risk Distribution")
    fig3 = px.histogram(
        filtered, x="PredictedRiskProb", nbins=20,
        labels={"PredictedRiskProb": "Predicted Attrition Probability"}
    )
    fig3.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Highest-Risk Employees in View")
    display_cols = ["JobRole", "Department", "OverTime", "JobSatisfaction",
                     "MonthlyIncome", "YearsAtCompany", "PredictedRiskProb"]
    top_risk = filtered.sort_values("PredictedRiskProb", ascending=False)[display_cols].head(10).copy()
    top_risk["PredictedRiskProb"] = top_risk["PredictedRiskProb"].map(lambda x: f"{x:.1%}")
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

# ---------------- TAB 3: RETENTION SIMULATOR ----------------
with tab3:
    st.subheader("Scenario Simulator")
    st.write(
        "Adjust the factors below to see how predicted attrition risk changes "
        "for a representative employee profile."
    )

    baseline = {}
    for col in feature_cols:
        if col in encoders:
            baseline[col] = df[col].mode()[0]
        else:
            baseline[col] = df[col].median()

    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Department", sorted(df["Department"].unique()), index=0)
        job_role = st.selectbox(
            "Job Role", sorted(df[df["Department"] == department]["JobRole"].unique())
        )
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        job_level = st.slider("Job Level", 1, 5, int(df["JobLevel"].median()))
        marital_status = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))

    with col2:
        monthly_income = st.slider(
            "Monthly Income ($)",
            int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max()),
            int(df["MonthlyIncome"].median()), step=100
        )
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, int(df["JobSatisfaction"].median()))
        work_life_balance = st.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, int(df["WorkLifeBalance"].median()))
        distance = st.slider(
            "Distance From Home (miles)",
            int(df["DistanceFromHome"].min()), int(df["DistanceFromHome"].max()),
            int(df["DistanceFromHome"].median())
        )
        years_at_company = st.slider(
            "Years at Company",
            int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max()),
            int(df["YearsAtCompany"].median())
        )

    scenario = baseline.copy()
    scenario.update({
        "Department": department,
        "JobRole": job_role,
        "OverTime": overtime,
        "JobLevel": job_level,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "DistanceFromHome": distance,
        "YearsAtCompany": years_at_company,
    })

    scenario_df = pd.DataFrame([scenario])[feature_cols]
    scenario_encoded = scenario_df.copy()
    for col, le in encoders.items():
        if col in scenario_encoded.columns:
            scenario_encoded[col] = le.transform(scenario_encoded[col].astype(str))

    risk_prob = model.predict_proba(scenario_encoded)[0, 1]

    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted Attrition Risk", f"{risk_prob:.1%}")
        if risk_prob >= 0.5:
            st.error("High risk profile")
        elif risk_prob >= 0.3:
            st.warning("Moderate risk profile")
        else:
            st.success("Low risk profile")

    with r2:
        st.subheader("What's Driving This Prediction")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scenario_encoded)

        if isinstance(shap_values, list):
            # Old format: list of arrays, one per class
            sv = shap_values[1][0]
        elif shap_values.ndim == 3:
            # New format: (samples, features, classes)
            sv = shap_values[0, :, 1]
        else:
            # Binary case already collapsed to (samples, features)
            sv = shap_values[0]

        sv = np.array(sv).flatten()

        shap_df = pd.DataFrame({"Feature": feature_cols, "Impact": sv})
        shap_df = shap_df.reindex(shap_df["Impact"].abs().sort_values(ascending=False).index).head(8)

        fig4 = px.bar(
            shap_df[::-1], x="Impact", y="Feature", orientation="h",
            color="Impact", color_continuous_scale="RdBu_r",
            labels={"Impact": "Contribution to Risk"}
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Recommended Actions")
    recs = []
    if scenario["OverTime"] == "Yes":
        recs.append("Reduce sustained overtime or add support staff. This is one of the strongest predictors of attrition in this model.")
    if scenario["JobSatisfaction"] <= 2:
        recs.append("Low job satisfaction is a major risk factor. A stay interview or role adjustment may help.")
    if scenario["WorkLifeBalance"] <= 2:
        recs.append("Poor work-life balance is strongly associated with attrition. Flexible scheduling may help.")
    if scenario["DistanceFromHome"] >= 20:
        recs.append("Long commute distance increases attrition risk. Hybrid or remote options could improve retention.")
    if not recs:
        recs.append("This profile doesn't show major red flags on the highest-impact factors. Risk here is likely tied to compensation or tenure, consider a market pay review.")

    for r in recs:
        st.write(f"- {r}")