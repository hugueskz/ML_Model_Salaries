import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Salary & Work Mode Oracle", page_icon="🤖", layout="centered")

st.title("🤖 The AI & Data Job Salary & Remote Oracle")
st.markdown("Welcome! Want to know if you are paid fairly or if you should be working in your pajamas from home? **Test the model below!** 🚀")

@st.cache_resource
def load_and_train_models():
    url = "https://raw.githubusercontent.com/gaspardcostes/projet-salaire-IA-ML/refs/heads/main/ai_jobs_salaries_clean.csv"
    df = pd.read_csv(url)
    
    # Cleaning and Outliers
    df = df.drop(columns=["isco_group_hint"], errors="ignore")
    Q1 = df['salary_in_usd'].quantile(0.25)
    Q3 = df['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['salary_in_usd'] >= Q1 - 1.5 * IQR) & (df['salary_in_usd'] <= Q3 + 1.5 * IQR)]
    
    # Feature Engineering
    df['location_grouped'] = df['company_location'].apply(lambda x: 'US' if x == 'US' else 'Outside_US')
    top_5_jobs = df['job_title'].value_counts().nlargest(5).index.tolist()
    df['job_title_grouped'] = df['job_title'].apply(lambda x: x if x in top_5_jobs else 'Other')
    df['work_mode_encoded'] = df['work_mode'].map({'On-site': 0, 'Hybrid': 1, 'Remote': 2})
    
    cat_features_adv = ['experience_level', 'employment_type', 'company_size', 'location_grouped', 'job_title_grouped']
    
    # --- Regression Model (Salary) ---
    features_reg_adv = cat_features_adv + ['work_year', 'remote_ratio']
    X_reg_adv = df[features_reg_adv]
    y_reg_adv = df['salary_in_usd']
    
    preprocessor_reg = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), cat_features_adv),
        remainder='passthrough'
    )
    model_reg = make_pipeline(preprocessor_reg, XGBRegressor(random_state=42, max_depth=6, learning_rate=0.1))
    model_reg.fit(X_reg_adv, y_reg_adv)
    
    # --- Classification Model (Work Mode) ---
    features_classif_adv = cat_features_adv + ['work_year', 'salary_in_usd']
    X_class_adv = df[features_classif_adv]
    y_class_adv = df['work_mode_encoded']
    
    preprocessor_class = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), cat_features_adv),
        remainder='passthrough'
    )
    model_class = make_pipeline(preprocessor_class, RandomForestClassifier(random_state=42, max_depth=10, class_weight='balanced'))
    model_class.fit(X_class_adv, y_class_adv)
    
    return model_reg, model_class, top_5_jobs

with st.spinner("🔮 The Oracle is consulting its data science grimoires... One moment!"):
    reg_model, class_model, top_jobs = load_and_train_models()

st.success("✨ The Oracle is ready!")

tab1, tab2 = st.tabs(["📊 Estimate My Salary", "🏡 Find My Work Mode"])

with tab1:
    st.header("💵 How much should you earn?")
    
    col1, col2 = st.columns(2)
    with col1:
        exp_mapping = {'Junior (EN)': 'EN', 'Mid-level (MI)': 'MI', 'Senior (SE)': 'SE', 'Executive/Director (EX)': 'EX'}
        exp_label = st.selectbox("Experience Level", list(exp_mapping.keys()), key="r_exp")
        exp_level = exp_mapping[exp_label]
        
        emp_mapping = {'Full-time': 'FT', 'Part-time': 'PT', 'Contractor': 'CT', 'Freelance': 'FL'}
        emp_label = st.selectbox("Employment Type", list(emp_mapping.keys()), key="r_emp")
        emp_type = emp_mapping[emp_label]
        
        comp_mapping = {'Small (< 50 employees)': 'S', 'Medium (50-250)': 'M', 'Large (> 250)': 'L'}
        comp_label = st.selectbox("Company Size", list(comp_mapping.keys()), key="r_size")
        comp_size = comp_mapping[comp_label]
        
    with col2:
        loc_mapping = {'🇺🇸 United States': 'US', '🌍 Rest of the World': 'Outside_US'}
        loc_label = st.selectbox("Location", list(loc_mapping.keys()), key="r_loc")
        loc = loc_mapping[loc_label]
        
        job = st.selectbox("Job Title", top_jobs + ['Other'], key="r_job")
        year = st.selectbox("Target Year", [2026, 2025, 2024], key="r_year")
        remote = st.slider("Estimated Remote Work Share (%)", 0, 100, 50, key="r_remote")

    st.markdown("---")
    if st.button("🚀 Predict Salary", type="primary"):
        input_data = pd.DataFrame([{
            'experience_level': exp_level,
            'employment_type': emp_type,
            'company_size': comp_size,
            'location_grouped': loc,
            'job_title_grouped': job,
            'work_year': year,
            'remote_ratio': remote
        }])
        prediction = reg_model.predict(input_data)[0]
        
        st.metric(label="💰 Estimated Annual Salary", value=f"${prediction:,.0f} USD")
        
        if prediction > 180000:
            st.balloons()
            st.success("🎉 Wow, that's a rockstar data salary!")
        elif prediction < 80000:
            st.info("💡 Interesting growth potential depending on market trends!")
        else:
            st.success("👍 That's a very solid compensation matching market averages.")

with tab2:
    st.header("💻 Office or Pajamas?")
    
    col1, col2 = st.columns(2)
    with col1:
        exp_label_c = st.selectbox("Experience Level", list(exp_mapping.keys()), key="c_exp")
        exp_level_c = exp_mapping[exp_label_c]
        
        emp_label_c = st.selectbox("Employment Type", list(emp_mapping.keys()), key="c_emp")
        emp_type_c = emp_mapping[emp_label_c]
        
        comp_label_c = st.selectbox("Company Size", list(comp_mapping.keys()), key="c_size")
        comp_size_c = comp_mapping[comp_label_c]
        
    with col2:
        loc_label_c = st.selectbox("Location", list(loc_mapping.keys()), key="c_loc")
        loc_c = loc_mapping[loc_label_c]
        
        job_c = st.selectbox("Job Title", top_jobs + ['Other'], key="c_job")
        year_c = st.selectbox("Target Year", [2026, 2025, 2024], key="c_year")
        salary_input = st.number_input("Target Salary (USD)", min_value=15000, max_value=500000, value=120000, step=5000, key="c_sal")

    st.markdown("---")
    if st.button("🔮 Predict Work Mode", type="primary"):
        input_data_c = pd.DataFrame([{
            'experience_level': exp_level_c,
            'employment_type': emp_type_c,
            'company_size': comp_size_c,
            'location_grouped': loc_c,
            'job_title_grouped': job_c,
            'work_year': year_c,
            'salary_in_usd': salary_input
        }])
        pred_code = class_model.predict(input_data_c)[0]
        
        mapping_inv = {
            0: '🏢 On-site (Get your suits and daily commute ready!)', 
            1: '🤝 Hybrid (The best of both worlds)', 
            2: '🏡 100% Remote (Time for pajamas and slippers)'
        }
        
        result_text = mapping_inv.get(pred_code, "Unknown")
        st.metric(label="🎯 Most Likely Work Mode", value=result_text)
