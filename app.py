from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.neet_model import (
    MODEL_PATH, METRICS_PATH, DISTRICT_CODES, SEX_CODES, RESIDENCE_CODES,
    EDUCATION_CODES, MARITAL_CODES,
)

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "district_indicators_2025_2023.csv"

st.set_page_config(page_title="Rwanda Opportunity Intelligence", page_icon="📊", layout="wide")

# Calm, accessible styling for a research-focused dashboard.
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 16px; background: linear-gradient(120deg,#3b2f2f,#80664f); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.25rem; margin-bottom: .3rem;}
.hero p {font-size: 1rem; margin-bottom: 0; opacity: .95;}
.note {padding: .85rem 1rem; background: #f7f3ed; border-left: 4px solid #80664f; border-radius: 6px; color: #322b26;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    numeric = [c for c in df.columns if c.endswith("_pct") or c.endswith("_pp") or c.endswith("_records") or c.endswith("_2023") or c.endswith("_sample_records")]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["district"] = df["district"].astype(str)
    return df

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the dashboard dataset: {exc}")
    st.stop()

required = {"district", "youth_neet_pct", "young_men_neet_pct", "young_women_neet_pct"}
if not required.issubset(data.columns):
    st.error("The district dataset is missing one or more required columns. Please check the CSV in the data folder.")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>Rwanda Opportunity Intelligence</h1>
  <p>Explore youth employment-related indicators across Rwanda's districts using official statistical datasets.</p>
</div>
""", unsafe_allow_html=True)
st.caption("A student-built analytical prototype for evidence-informed discussion. It is not an official NISR product.")

with st.sidebar:
    st.header("Explore the data")
    page = st.radio("Dashboard section", ["National overview", "District explorer", "Gender comparison", "NEET prediction lab", "Economic context", "Methodology & sources"])
    district_options = sorted(data["district"].dropna().unique().tolist())
    selected_district = st.selectbox("District focus", ["All districts"] + district_options)
    st.divider()
    st.caption("Labour Force Survey: 2025\n\nEstablishment Census: 2023")

filtered = data if selected_district == "All districts" else data[data["district"] == selected_district]

if page == "National overview":
    st.subheader("National labour-market snapshot")
    st.markdown("The headline national values below are from the project's earlier analysis of the NISR 2025 Labour Force Survey. They should be rechecked against the exact official table and indicator definitions before final publication.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Youth NEET (ages 16–30)", "24.5%", help="Previously calculated weighted estimate; verify definition against NISR report.")
    k2.metric("Youth unemployment", "14.7%", help="Previously calculated weighted estimate; verify indicator definition against NISR report.")
    k3.metric("Young women NEET", "30.5%", help="Previously calculated weighted estimate.")
    k4.metric("Young men NEET", "18.3%", help="Previously calculated weighted estimate.")
    st.markdown('<div class="note"><b>Key descriptive finding:</b> the previous national estimates suggest a female-minus-male youth NEET gap of about 12.2 percentage points. This is a descriptive difference, not a causal explanation.</div>', unsafe_allow_html=True)
    st.subheader("District youth NEET estimates")
    chart = data.sort_values("youth_neet_pct", ascending=True)
    fig = px.bar(chart, x="youth_neet_pct", y="district", orientation="h", labels={"youth_neet_pct":"Youth NEET estimate (%)", "district":"District"}, hover_data=["youth_neet_ci_lower_pct", "youth_neet_ci_upper_pct", "youth_neet_sample_records"], title="Estimated share of youth not in employment, education, or training")
    fig.update_layout(height=850, yaxis_title="", xaxis_title="Youth NEET estimate (%)", margin=dict(l=10,r=20,t=65,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("District estimates and intervals are from the existing project analysis. Confidence intervals are provisional pending verification of the survey variance method.")
    st.subheader("District data preview")
    st.dataframe(data[[c for c in ["district", "youth_neet_pct", "youth_neet_ci_lower_pct", "youth_neet_ci_upper_pct", "young_men_neet_pct", "young_women_neet_pct", "neet_gender_gap_pp_women_minus_men"] if c in data.columns]].sort_values("youth_neet_pct", ascending=False).round(2), use_container_width=True, hide_index=True)

elif page == "District explorer":
    st.subheader("District opportunity explorer")
    if selected_district == "All districts":
        view = data.sort_values("youth_neet_pct", ascending=False)
        st.write("Select a district in the sidebar to focus on one area, or compare all districts below.")
        metric = st.selectbox("Rank districts by", ["Youth NEET (%)", "Youth unemployment (%)", "Employment-to-population (%)", "Labour-force participation (%)"])
        metric_map = {"Youth NEET (%)":"youth_neet_pct", "Youth unemployment (%)":"youth_unemployment_pct", "Employment-to-population (%)":"employment_to_population_pct", "Labour-force participation (%)":"labour_force_participation_pct"}
        col = metric_map[metric]
        if col in data.columns:
            view = data.sort_values(col, ascending=False)
            fig = px.bar(view, x=col, y="district", orientation="h", labels={col:metric, "district":"District"}, title=f"District comparison: {metric}")
            fig.update_layout(height=850, yaxis_title="", margin=dict(l=10,r=20,t=65,b=20))
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        row = filtered.iloc[0]
        st.markdown(f"### {row['district']}")
        a,b,c = st.columns(3)
        a.metric("Youth NEET", f"{row['youth_neet_pct']:.1f}%")
        b.metric("Youth unemployment", f"{row['youth_unemployment_pct']:.1f}%" if pd.notna(row.get("youth_unemployment_pct")) else "N/A")
        c.metric("Labour-force participation", f"{row['labour_force_participation_pct']:.1f}%" if pd.notna(row.get("labour_force_participation_pct")) else "N/A")
        if pd.notna(row.get("youth_neet_ci_lower_pct")) and pd.notna(row.get("youth_neet_ci_upper_pct")):
            st.info(f"Provisional 95% confidence interval for youth NEET: {row['youth_neet_ci_lower_pct']:.1f}%–{row['youth_neet_ci_upper_pct']:.1f}%. Sample records: {int(row['youth_neet_sample_records']) if pd.notna(row.get('youth_neet_sample_records')) else 'not available'}.")
        sex = pd.DataFrame({"Group":["Young men", "Young women"], "NEET estimate (%)":[row["young_men_neet_pct"],row["young_women_neet_pct"]]})
        fig = px.bar(sex, x="Group", y="NEET estimate (%)", color="Group", title=f"Youth NEET by sex — {row['district']}", text_auto=".1f")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered.T.rename(columns={filtered.index[0]:"Value"}), use_container_width=True)

elif page == "Gender comparison":
    st.subheader("Youth NEET by sex")
    st.write("Compare point estimates for young men and young women across districts. The gap is young women minus young men, measured in percentage points.")
    g = data[["district", "young_men_neet_pct", "young_women_neet_pct", "neet_gender_gap_pp_women_minus_men"]].copy()
    g = g.sort_values("neet_gender_gap_pp_women_minus_men", ascending=False)
    long = g.melt(id_vars="district", value_vars=["young_men_neet_pct", "young_women_neet_pct"], var_name="Group", value_name="NEET estimate (%)")
    long["Group"] = long["Group"].map({"young_men_neet_pct":"Young men", "young_women_neet_pct":"Young women"})
    fig = px.bar(long, x="NEET estimate (%)", y="district", color="Group", barmode="group", orientation="h", title="Youth NEET estimates by sex and district")
    fig.update_layout(height=850, yaxis_title="", margin=dict(l=10,r=20,t=65,b=20))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(g, x="neet_gender_gap_pp_women_minus_men", y="district", orientation="h", title="Female-minus-male NEET gap (percentage points)", labels={"neet_gender_gap_pp_women_minus_men":"Gap (percentage points)","district":"District"})
    fig2.update_layout(height=850, yaxis_title="", margin=dict(l=10,r=20,t=65,b=20))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(g.round(2), use_container_width=True, hide_index=True)
    st.warning("The gender gap is descriptive. This prototype does not estimate a design-based confidence interval for the difference, so do not label district gender gaps statistically significant.")

elif page == "Economic context":
    st.subheader("Establishment Census context (2023)")
    st.markdown("These figures describe establishments, not vacancies or the number of employed people. They are shown separately from the 2025 Labour Force Survey because the sources measure different things and refer to different years.")
    establishment_cols = [c for c in ["working_establishment_records_2023", "formal_establishment_records_2023", "informal_establishment_records_2023", "all_establishment_records_2023", "temporarily_closed_records_2023"] if c in data.columns]
    if establishment_cols:
        label_map = {"working_establishment_records_2023":"Working establishments (records)","formal_establishment_records_2023":"Formal establishments (records)","informal_establishment_records_2023":"Informal establishments (records)","all_establishment_records_2023":"All establishments (records)","temporarily_closed_records_2023":"Temporarily closed (records)"}
        selected_measure = st.selectbox("Measure", establishment_cols, format_func=lambda x: label_map.get(x,x))
        fig = px.bar(data.sort_values(selected_measure, ascending=False), x=selected_measure, y="district", orientation="h", title=label_map.get(selected_measure,selected_measure), labels={selected_measure:"Establishment records", "district":"District"})
        fig.update_layout(height=850, yaxis_title="", margin=dict(l=10,r=20,t=65,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(data[["district"]+establishment_cols].sort_values(selected_measure, ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Establishment indicators are not available in the loaded dataset.")
    st.caption("Confirm source definitions and public-use terms before publishing these derived aggregates.")

elif page == "NEET prediction lab":
    st.subheader("Youth NEET prediction lab")
    st.write(
        "Explore how a transparent statistical model estimates NEET likelihood for a hypothetical youth profile. "
        "This is a research demonstration, not a decision tool for judging or excluding individuals."
    )
    st.warning(
        "A model artifact must be trained locally from the authorized LFS 2025 microdata. "
        "Do not upload raw microdata to GitHub. Before publishing a trained model file, check the NISR data-access terms."
    )
    if not MODEL_PATH.exists():
        st.info("The model has not been trained in this project folder yet.")
        st.markdown("""
**Train the model on your computer**

1. Keep the LFS file private on your computer.
2. In PowerShell, from the project folder, install the updated dependencies with `pip install -r requirements.txt`.
3. Run the training command shown in the README, replacing the path with your actual LFS file path.
4. Review the evaluation metrics printed by the script before using or publishing the model.

The training script holds out entire primary sampling units (PSUs) for testing and uses `weight2` for model fitting and weighted metrics. This is a practical prototype, not a full complex-survey variance analysis.
""")
    else:
        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle["model"]
            metrics = bundle.get("metrics", {})
            if METRICS_PATH.exists():
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            st.error(f"The model artifact could not be loaded: {exc}")
            st.stop()

        st.markdown("#### Held-out test performance")
        st.caption("Metrics come from the locally trained PSU-level holdout. They are not official NISR estimates and should be interpreted with the model's limitations. The baseline model's performance is exploratory, not sufficient on its own for operational targeting.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Weighted ROC-AUC", f"{metrics.get('roc_auc_weighted', float('nan')):.3f}")
        metric_cols[1].metric("Weighted PR-AUC", f"{metrics.get('average_precision_weighted', float('nan')):.3f}")
        metric_cols[2].metric("Weighted recall", f"{metrics.get('recall_weighted', float('nan')):.3f}")
        metric_cols[3].metric("Weighted precision", f"{metrics.get('precision_weighted', float('nan')):.3f}")
        st.caption(
            f"Training records: {metrics.get('training_records', 'N/A'):,} | "
            f"Test records: {metrics.get('test_records', 'N/A'):,} | "
            f"Test PSUs: {metrics.get('test_psus', 'N/A'):,} | "
            f"Weighted test NEET rate: {metrics.get('test_weighted_neet_rate_pct', 'N/A')}%"
        )
        with st.expander("Full evaluation details and limitations"):
            st.json(metrics)

        st.markdown("#### Try a hypothetical profile")
        st.caption("The estimate reflects associations learned from the 2025 survey sample. It is not a diagnosis, a guarantee about a person, or evidence of causation.")
        district_names = sorted(DISTRICT_CODES.keys())
        form1, form2, form3 = st.columns(3)
        with form1:
            profile_district = st.selectbox("District", district_names, key="pred_district")
            profile_age = st.slider("Age", min_value=16, max_value=30, value=22, key="pred_age")
            profile_sex = st.selectbox("Sex", list(SEX_CODES.keys()), key="pred_sex")
        with form2:
            profile_residence = st.selectbox("Area of residence", list(RESIDENCE_CODES.keys()), key="pred_residence")
            profile_education = st.selectbox("Highest education completed", list(EDUCATION_CODES.keys()), index=1, key="pred_education")
            profile_marital = st.selectbox("Marital status", list(MARITAL_CODES.keys()), index=5, key="pred_marital")
        with form3:
            profile_disability = st.selectbox("Reported disability indicator", ["No", "Yes"], key="pred_disability")
            profile_hhsize = st.slider("Household size", min_value=1, max_value=15, value=5, key="pred_hhsize")
            threshold = st.slider("Alert threshold", min_value=0.10, max_value=0.80, value=float(metrics.get("threshold", 0.50)), step=0.01, key="pred_threshold")

        if st.button("Estimate hypothetical profile", type="primary"):
            profile = pd.DataFrame([{
                "A01": str(SEX_CODES[profile_sex]),
                "A04": profile_age,
                "A05": str(MARITAL_CODES[profile_marital]),
                "Code_UR": str(RESIDENCE_CODES[profile_residence]),
                "code_dis": str(DISTRICT_CODES[profile_district]),
                "attained": str(EDUCATION_CODES[profile_education]),
                "disable": str(1 if profile_disability == "Yes" else 0),
                "hhsize": profile_hhsize,
            }])
            probability = float(model.predict_proba(profile)[0, 1])
            st.metric("Model-estimated NEET likelihood", f"{probability:.1%}")
            if probability >= threshold:
                st.warning("This hypothetical profile is above your selected threshold. Treat this only as a prompt to examine aggregate evidence and support options—not as a decision about an individual.")
            else:
                st.info("This hypothetical profile is below your selected threshold. A lower model score does not mean there is no need for support.")
            st.caption("Do not use this score to deny services, rank real individuals, or make high-stakes decisions. Model quality and group-level fairness must be reviewed before any real-world use.")

else:
    st.subheader("Methodology, sources, and limitations")
    st.markdown("""
### Research question
How do employment-related outcomes vary across Rwanda's districts and population groups, and where might the data support further investigation?

### Indicators shown
- **Youth NEET:** estimated percentage of youth not in employment, education, or training. The existing analysis uses ages 16–30; verify the exact operational definition against the NISR report and variable dictionary.
- **Youth unemployment:** a labour-force indicator; it is not interchangeable with NEET.
- **Employment-to-population ratio** and **labour-force participation:** included where available in the prepared district dataset.
- **Gender gap:** young-women NEET estimate minus young-men NEET estimate, expressed in percentage points.
- **Establishment counts:** 2023 establishment records, presented only as economic context, not vacancies or counts of workers.

### Sources
- National Institute of Statistics of Rwanda (NISR), *Annual Labour Force Survey 2025 Report*.
- NISR, Establishment Census 2023 dataset and accompanying documentation.
- Project-prepared aggregate district indicators, derived from existing analysis outputs.

### Limitations
1. District estimates are descriptive and do not establish causes.
2. The confidence intervals in the prepared dataset are provisional. Verify the variance-estimation method, strata, primary sampling units, weights, and degrees of freedom against official survey documentation before policy use.
3. Do not infer statistical significance from district ranks or overlapping confidence intervals.
4. LFS 2025 and Establishment Census 2023 are different sources, years, and units of analysis.
5. Confirm permission to publish derived aggregates. Never upload raw or restricted microdata to a public repository.
6. This is a student prototype, not an official NISR product. Validate all figures against source tables before submission.

### Prediction methodology
- Target: `youngs == 4` among records where `young == 1`, based on the supplied LFS variable labels (youth aged 16–30; NEET status).
- Predictors: sex, age, marital status, urban/rural residence, district, highest education completed, disability indicator, and household size.
- Excluded: direct employment status, current education/training status, and derived NEET indicators to reduce target leakage.
- Model: weighted logistic regression using annual survey weight `weight2`.
- Evaluation: hold out entire primary sampling units (PSUs), tune the classification threshold on a separate PSU-level validation split, and report weighted ROC-AUC, average precision, precision, recall, F1 and Brier score on the untouched test split.
- Limitation: PSU grouping and use of annual weights do not replace full complex-survey variance estimation. The model is exploratory, has only one survey year, and does not forecast future district unemployment. Its current performance is modest and it must not be used to deny services or make high-stakes decisions.

### AI-use disclosure template
"We used an AI assistant to support software scaffolding, code drafting, and documentation. Team members reviewed and tested the application and are responsible for validating all data, calculations, sources, and final claims." Edit this statement so it accurately reflects your team's actual use.
""")
    st.subheader("Download the displayed district data")
    st.download_button("Download district indicators (CSV)", data=data.to_csv(index=False).encode("utf-8"), file_name="rwanda_opportunity_district_indicators.csv", mime="text/csv")

st.divider()
st.caption("Rwanda Opportunity Intelligence • Hackathon prototype • Check definitions and permissions before public release")
