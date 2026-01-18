import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import requests
import os
import pickle

# === Функции для скачивания с Google Drive ===
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# === Настройки моделей ===
MODEL_IDS = {
    "credit_scoring_model.pkl": "1Vv0mbisLkITeQ5k39cV0VQ3pz7gCsCfg",
    "log_reg_explain.pkl": "1IB2dWaEUVfp3HO8diqdsyFT6qF-cVW4x"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for model_name, file_id in MODEL_IDS.items():
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"📥 Скачиваем {model_name} из Google Drive...")
        download_file_from_google_drive(file_id, model_path)
        print(f"✅ {model_name} загружен")
    else:
        print(f"✅ {model_name} уже существует — пропускаем загрузку")
     
st.set_page_config(
    page_title="PD Credit Scoring",
    page_icon="💳",
    layout="wide"
)

 
 
num_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length"
]

cat_cols = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

feature_names = (
    num_cols +
    list(
        log_reg.named_steps["prep"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
    )
)

FEATURE_DESCRIPTIONS = {
    "person_age": "Возраст клиента",
    "person_income": "Годовой доход",
    "person_emp_length": "Стаж работы",
    "loan_amnt": "Сумма кредита",
    "loan_int_rate": "Процентная ставка",
    "cb_person_cred_hist_length": "Длина кредитной истории",
    "person_home_ownership_RENT": "Жильё в аренде",
    "person_home_ownership_OWN": "Собственное жильё",
    "person_home_ownership_MORTGAGE": "Жильё в ипотеке",
    "loan_intent_PERSONAL": "Кредит на личные нужды",
    "loan_intent_EDUCATION": "Кредит на образование",
    "loan_intent_MEDICAL": "Кредит на медицинские расходы",
    "loan_intent_VENTURE": "Кредит на бизнес",
    "loan_intent_DEBTCONSOLIDATION": "Консолидация долгов",
    "loan_intent_HOMEIMPROVEMENT": "Ремонт дома",
    "cb_person_default_on_file_Y": "Были просрочки в прошлом",
    "cb_person_default_on_file_N": "Просрочек в прошлом не было",
}

def human_feature_name(f):
    return FEATURE_DESCRIPTIONS.get(f, f.replace("_", " "))
 
st.markdown("""
<style>
    .block {
        background-color: white;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        margin-bottom: 25px;
    }
    .title {
        font-size: 40px;
        font-weight: 800;
    }
    .subtitle {
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)
 
st.markdown('<div class="title">💳 Оценка вероятности дефолта (PD)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Модель оценки кредитного риска по потребительским кредитам</div>',
    unsafe_allow_html=True
)
 
tab1, tab2, tab3 = st.tabs(["💳 Скоринг", "📊 Данные", "🧠 Модель"])
 
with tab1:
    st.sidebar.header("👤 Данные клиента")

    age = st.sidebar.slider("Возраст", 18, 75, 30)
    income = st.sidebar.number_input("Годовой доход", 0, value=50000, step=1000)
    emp_length = st.sidebar.slider("Стаж работы (лет)", 0, 40, 5)
    cred_hist = st.sidebar.slider("Кредитная история (лет)", 0, 30, 3)

    home = st.sidebar.selectbox(
        "Тип жилья",
        ["АРЕНДА", "СОБСТВЕННОСТЬ", "ИПОТЕКА"]
    )

    prev_default = st.sidebar.selectbox(
        "Просрочки в прошлом",
        ["НЕТ", "ДА"]
    )
    
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("💼 Параметры кредита")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amount = st.number_input("Сумма кредита", 0, value=10000, step=500)

    with col2:
        loan_rate = st.slider("Процентная ставка (%)", 5.0, 40.0, 12.0)

    loan_intent = st.selectbox(
        "Цель кредита",
        [
            "ЛИЧНЫЕ НУЖДЫ",
            "ОБРАЗОВАНИЕ",
            "МЕДИЦИНА",
            "БИЗНЕС",
            "КОНСОЛИДАЦИЯ ДОЛГОВ",
            "РЕМОНТ ДОМА"
        ]
    )
    

    st.markdown('</div>', unsafe_allow_html=True)

    def explain_prediction(log_reg, input_df, feature_names, top_n=5):
        X_transformed = log_reg.named_steps["prep"].transform(input_df)

        X_dense = (
            X_transformed.toarray()
            if hasattr(X_transformed, "toarray")
            else X_transformed
        )

        contributions = X_dense[0] * log_reg.named_steps["model"].coef_[0]

        expl_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions
        })

        expl_df["abs"] = expl_df["contribution"].abs()
        expl_df = expl_df.sort_values("abs", ascending=False)

        return expl_df.head(top_n)

    if st.button("🔍 Рассчитать вероятность дефолта", use_container_width=True):

        input_df = pd.DataFrame([{
            "person_age": age,
            "person_income": income,
            "person_home_ownership": {
                "АРЕНДА": "RENT",
                "СОБСТВЕННОСТЬ": "OWN",
                "ИПОТЕКА": "MORTGAGE"
            }[home],
            "person_emp_length": emp_length,
            "loan_intent": {
                "ЛИЧНЫЕ НУЖДЫ": "PERSONAL",
                "ОБРАЗОВАНИЕ": "EDUCATION",
                "МЕДИЦИНА": "MEDICAL",
                "БИЗНЕС": "VENTURE",
                "КОНСОЛИДАЦИЯ ДОЛГОВ": "DEBTCONSOLIDATION",
                "РЕМОНТ ДОМА": "HOMEIMPROVEMENT"
            }[loan_intent], 
            "loan_amnt": loan_amount,
            "loan_int_rate": loan_rate, 
            "cb_person_default_on_file": "Y" if prev_default == "ДА" else "N",
            "cb_person_cred_hist_length": cred_hist
        }])

        pd_value = model.predict_proba(input_df)[0][1]

        explanation = explain_prediction(
            log_reg,
            input_df,
            feature_names,
            top_n=5
        )
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📊 Результат скоринга")

        st.metric("Вероятность дефолта (PD)", f"{pd_value:.2%}")
        st.progress(float(pd_value))

        if pd_value < 0.3:
            st.success("🟢 Низкий риск — рекомендуется одобрение кредита")
        elif pd_value < 0.6:
            st.warning("🟡 Средний риск — одобрение с дополнительными условиями")
        else:
            st.error("🔴 Высокий риск — рекомендуется отказ")

        st.subheader("📌 Почему получился такой результат")

        risk_up = explanation[explanation["contribution"] > 0]
        risk_down = explanation[explanation["contribution"] < 0]

        if not risk_up.empty:
            st.markdown("**Факторы, увеличившие риск:**")
            for f in risk_up["feature"]:
                st.markdown(f"- {human_feature_name(f)}")


        if not risk_down.empty:
            st.markdown("**Факторы, снизившие риск:**")
            for f in risk_down["feature"]:
                st.markdown(f"- {human_feature_name(f)}") 

        st.markdown("""
        **Что означает PD?**  
        Вероятность дефолта (Probability of Default) показывает риск того,
        что клиент не выполнит свои кредитные обязательства.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("ℹ️ О модели"):
            st.write("""
            **Тип модели:** Логистическая регрессия  
            **Задача:** Оценка PD  
            **Метрики:** ROC-AUC, Gini  
            **Использование:** Учебный и демонстрационный проект  
            """) 

with tab2:
    @st.cache_data
    def load_data():
        data_path = os.path.join(BASE_DIR, "..", "data", "credit_dataset.csv")
        return pd.read_csv(data_path, sep=";")


    df = load_data()

    st.subheader("📄 Обучающие данные")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("🎯 Распределение дефолтов")
    st.bar_chart(df["loan_status"].value_counts(normalize=True))
 
with tab3:
    coef = log_reg.named_steps["model"].coef_[0]

    imp_df = pd.DataFrame({
        "Признак": [human_feature_name(f) for f in feature_names],
        "Влияние": coef
    }).sort_values("Влияние", key=abs, ascending=False)

    st.subheader("📊 Влияние признаков")
    st.bar_chart(imp_df.head(10).set_index("Признак"))

st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Credit Risk Scoring · Probability of Default · Streamlit
</p>
""", unsafe_allow_html=True)
