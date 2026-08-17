import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------
st.set_page_config(page_title="Lead & Income Predictor", page_icon="📊", layout="centered")
st.title("📊 Lead & Income Predictor")
st.caption("Predicts Leads, Sum Income, and Sum Scrub Cost using a Gradient Boosting model "
           "trained on historical partner/offer performance data.")

# ----------------------------------------------------------------
# Load model and supporting data (cached so it only loads once)
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("gbm_pipeline.pkl")
    partners = joblib.load("known_partners.pkl")
    offers = joblib.load("known_offers.pkl")
    return model, partners, offers

model, known_partners, known_offers = load_model()

features_raw = ['Sum Cost', 'Partner', 'Offer', 'Month', 'Year']
target = ['Leads', 'Sum Income', 'Sum Scrub Cost']

# Model-level performance stats, from final notebook results
model_stats = {
    'Leads':          {'R2': 0.950, 'MAE': 16.90},
    'Sum Income':     {'R2': 0.796, 'MAE': 131.78},
    'Sum Scrub Cost': {'R2': 0.979, 'MAE': 6.40},
}

# ----------------------------------------------------------------
# Input form
# ----------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        partner = st.selectbox("Partner", known_partners)
        offer = st.selectbox("Offer", known_offers)

    with col2:
        sum_cost = st.number_input("Sum Cost ($)", min_value=0.0, value=1000.0, step=50.0)
        month = st.selectbox("Month", list(range(1, 13)), index=5)
        year = st.number_input("Year", min_value=2022, max_value=2030, value=2026, step=1)

    submitted = st.form_submit_button("Predict")

# ----------------------------------------------------------------
# Prediction logic
# ----------------------------------------------------------------
if submitted:

    # --- Data sufficiency / extrapolation warnings ---
    warnings = []

    input_df = pd.DataFrame([{
        'Sum Cost': sum_cost,
        'Partner': partner,
        'Offer': offer,
        'Month': month,
        'Year': year
    }])[features_raw]

    pred = model.predict(input_df)[0]
    raw_result = dict(zip(target, pred))

    if raw_result['Leads'] < 0:
        warnings.append(
            f"Model produced a negative Leads prediction ({raw_result['Leads']:.1f}), "
            f"clipped to 0. This usually means the requested Sum Cost is far outside "
            f"the historical range for this Partner/Offer — treat this result as unreliable."
        )

    leads = max(round(raw_result['Leads']), 0)
    income = round(raw_result['Sum Income'], 2)
    scrub = round(raw_result['Sum Scrub Cost'], 2)

    income_per_lead = round(income / leads, 2) if leads > 0 else 0.0
    scrub_per_lead = round(scrub / leads, 2) if leads > 0 else 0.0

    if warnings:
        for w in warnings:
            st.warning(w)

    st.subheader("Prediction Results")

    summary = pd.DataFrame([
        {'Metric': 'Partner', 'Value': partner, 'Model Confidence': ''},
        {'Metric': 'Offer', 'Value': offer, 'Model Confidence': ''},
        {'Metric': 'Month/Year', 'Value': f"{month:02d}/{year}", 'Model Confidence': ''},
        {'Metric': 'Input Sum Cost', 'Value': f"${sum_cost:,.2f}", 'Model Confidence': ''},
        {'Metric': 'Predicted Leads', 'Value': f"{leads:,}",
         'Model Confidence': f"~{model_stats['Leads']['R2']*100:.0f}% (avg error ±{model_stats['Leads']['MAE']:.0f} leads)"},
        {'Metric': 'Predicted Sum Income', 'Value': f"${income:,.2f}",
         'Model Confidence': f"~{model_stats['Sum Income']['R2']*100:.0f}% (avg error ±${model_stats['Sum Income']['MAE']:.2f})"},
        {'Metric': 'Predicted Sum Scrub Cost', 'Value': f"${scrub:,.2f}",
         'Model Confidence': f"~{model_stats['Sum Scrub Cost']['R2']*100:.0f}% (avg error ±${model_stats['Sum Scrub Cost']['MAE']:.2f})"},
        {'Metric': 'Income per Lead', 'Value': f"${income_per_lead:,.2f}", 'Model Confidence': ''},
        {'Metric': 'Scrub Cost per Lead', 'Value': f"${scrub_per_lead:,.2f}", 'Model Confidence': ''},
    ])

    st.table(summary.set_index('Metric'))
    st.caption("Model Confidence reflects historical model performance (R² and average error), "
               "not certainty about this specific prediction. Predictions for Partner/Offer "
               "combinations with limited historical data, or spend levels far outside past "
               "experience, are less reliable.")
