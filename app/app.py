import json as encoder
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(42)
import datetime

st.set_page_config(
    page_title="PD Credit Scoring",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "..", "data", "credit_dataset.csv")
    return pd.read_csv(data_path, sep=";")

df = load_data()

start_date = pd.to_datetime("2021-01-01")
end_date = pd.to_datetime("2024-12-31")

df["loan_date"] = start_date + pd.to_timedelta(
    np.random.randint(0, (end_date - start_date).days, size=len(df)),
    unit="D"
)

df["month"] = df["loan_date"].dt.month
df["quarter"] = df["loan_date"].dt.quarter


model = joblib.load(os.path.join(BASE_DIR, "credit_scoring_model.pkl"))
log_reg = joblib.load(os.path.join(BASE_DIR, "log_reg_explain.pkl"))
knn = joblib.load(os.path.join(BASE_DIR, "knn_recommender.pkl"))
knn_scaler = joblib.load(os.path.join(BASE_DIR, "knn_scaler.pkl"))

 
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

encoder = log_reg.named_steps["prep"].named_transformers_["cat"]

feature_names = log_reg.named_steps["prep"].get_feature_names_out() 
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

def knn_recommend(input_df, df, knn, scaler, k):
    reco_features = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "cb_person_cred_hist_length"
    ] 
    x = scaler.transform(input_df[reco_features])
    _, idx = knn.kneighbors(x, n_neighbors=k)
    neighbors = df.iloc[idx[0]] 
    neighbors = neighbors.dropna(subset=["loan_status"]) 
    default_rate = neighbors["loan_status"].mean()
    avg_amount = neighbors["loan_amnt"].mean()
    avg_rate = neighbors["loan_int_rate"].mean() 
    if default_rate < 0.3:
        decision = "Одобрить"
    elif default_rate < 0.6:
        decision = "Одобрить с условиями"
    else:
        decision = "Отказать" 
    neighbors_count = len(neighbors) 
    return decision, default_rate, avg_amount, avg_rate, neighbors_count, neighbors


def explain_knn(input_df, neighbors):
    reasons = []
    client = input_df.iloc[0]
    def compare(feature, label, higher_is_risk=True, threshold=0.15):
        neigh_mean = neighbors[feature].mean()
        diff = (client[feature] - neigh_mean) / neigh_mean

        if higher_is_risk and diff > threshold:
            reasons.append(f"{label} выше среднего по похожим клиентам")
        elif not higher_is_risk and diff < -threshold:
            reasons.append(f"{label} ниже среднего по похожим клиентам")

    compare(
        feature="loan_amnt",
        label="Запрашиваемая сумма",
        higher_is_risk=True
    )

    compare(
        feature="loan_int_rate",
        label="Процентная ставка",
        higher_is_risk=True
    )

    compare(
        feature="cb_person_cred_hist_length",
        label="Длина кредитной истории",
        higher_is_risk=False
    )

    compare(
        feature="person_income",
        label="Доход клиента",
        higher_is_risk=False
    )

    if not reasons:
        reasons.append("Параметры клиента близки к средним значениям по группе")

    return reasons


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
 
st.markdown("""
    <div style="
    background: linear-gradient(90deg,#0f172a,#020617);
    padding:35px;
    border-radius:18px;
    margin-bottom:25px">

    <h1 style='color:white;font-size:42px;margin-bottom:10px'>
    🏦 AI-система кредитного скоринга и рекомендаций
    </h1>

    <p style='color:#cbd5f5;font-size:18px;margin-bottom:8px'>
    Оценка вероятности дефолта • Рекомендация кредита • Сезонная аналитика • Risk dashboard
    </p>

    <p style='color:#94a3b8;font-size:14px'>
    Интеллектуальная система поддержки кредитных решений на основе машинного обучения
    </p>

    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="subtitle">Модель оценки кредитного риска по потребительским кредитам</div>',
    unsafe_allow_html=True
)
 
tab1, tab_reco, tab2, tab3 = st.tabs([
    "💳 Скоринг",
    "🎯 Рекомендации",
    "📊 Данные",
    "🧠 Модель"
])

 
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

    if st.button("🔍 Рассчитать вероятность дефолта", width="stretch"):
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
        
        today = datetime.datetime.today()
 
        input_df["month"] = today.month
 
        input_df["quarter"] = (today.month - 1)//3 + 1
 
        def get_season(m):
            if m in [12,1,2]:
                return "winter"
            elif m in [3,4,5]:
                return "spring"
            elif m in [6,7,8]:
                return "summer"
            else:
                return "autumn"

        input_df["season"] = get_season(today.month)

        pd_value = model.predict_proba(input_df)[0][1] 
        st.session_state["last_input"] = input_df
        st.session_state["last_pd"] = pd_value
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

with tab_reco:
    st.markdown("""
        <div style="
        background: linear-gradient(90deg,#020617,#111827);
        padding:25px;
        border-radius:16px;
        margin-bottom:18px;
        box-shadow:0 6px 20px rgba(0,0,0,0.25);
        ">

        <h2 style='color:#f8fafc;margin-bottom:6px'>
        🤖 Интеллектуальная система рекомендаций по кредиту
        </h2>

        <p style='color:#cbd5e1;font-size:15px;margin-bottom:4px'>
        Анализ похожих клиентов с помощью алгоритма K-ближайших соседей (KNN)
        </p>

        <p style='color:#94a3b8;font-size:13px'>
        Система оценивает риск группы и предлагает решение по выдаче кредита
        </p>

        </div>
        """, unsafe_allow_html=True)


 
    k_user = st.slider(
        "🔧 Количество похожих клиентов (K)",
        min_value=5,
        max_value=30,
        value=10,
        step=1
    )

    auto_k = int(len(df) ** 0.5)
    st.caption(f"🤖 Рекомендуемое K по данным: ~{auto_k}")

    if k_user < 8:
        st.warning("⚠️ Малое K — рекомендация может быть нестабильной")
    elif k_user > 20:
        st.info("ℹ️ Большое K — рекомендация более сглаженная, но менее чувствительная")
    else:
        st.success("✅ Оптимальный диапазон K для стабильной рекомендации")
 
    if "last_input" not in st.session_state:
        st.info("ℹ️ Сначала рассчитайте скоринг клиента во вкладке «💳 Скоринг»")
    else:
        input_df = st.session_state["last_input"]
 
        decision, neigh_pd, rec_amount, rec_rate, neighbors_count, neighbors = knn_recommend(
            input_df=input_df,
            df=df,
            knn=knn,
            scaler=knn_scaler,
            k=k_user
        )

        reasons = explain_knn(input_df, neighbors)

        st.markdown('<div class="block">', unsafe_allow_html=True)
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Похожие клиенты", neighbors_count)
        c2.metric("⚠️ Риск группы", f"{neigh_pd:.1%}")
        c3.metric("💰 Сумма", f"{rec_amount:,.0f}")
        c4.metric("📈 Ставка", f"{rec_rate:.1f}%")
 
        confidence = 1 - neighbors["loan_status"].std()
        confidence = max(0, min(confidence, 1))

        st.progress(float(confidence))
        st.caption(f"📊 Надёжность рекомендации: {confidence:.0%}")

        if confidence < 0.4:
            st.warning("⚠️ Низкая надёжность — клиенты слишком разные")
        elif confidence < 0.7:
            st.info("ℹ️ Средняя надёжность рекомендации")
        else:
            st.success("✅ Высокая надёжность рекомендации")
 
        st.subheader("🏦 Рекомендация системы")

        if decision == "Одобрить":
            st.success("✅ APPROVED — низкий риск клиента")
        elif decision == "Одобрить с условиями":
            st.warning("⚠️ CONDITIONAL APPROVAL — средний риск")
        else:
            st.error("❌ REJECTED — высокий риск")

 
        st.markdown("### 📌 Почему такая рекомендация")

        for r in reasons:
            r_low = r.lower()

            if "доход" in r_low:
                color, emoji = "#22c55e", "💰"
            elif "ставк" in r_low or "процент" in r_low:
                color, emoji = "#38bdf8", "📈"
            elif "истор" in r_low:
                color, emoji = "#facc15", "📜"
            else:
                color, emoji = "#a855f7", "🔹"

            st.markdown(
                f"""
                <div style="
                    background:#020617;
                    color:#e5e7eb;
                    padding:14px 16px;
                    border-radius:12px;
                    margin-bottom:10px;
                    border-left:4px solid {color};
                    font-size:15px;
                ">
                {emoji} {r}
                </div>
                """,
                unsafe_allow_html=True
            )
 
        if "last_pd" in st.session_state:
            pd_model = st.session_state["last_pd"]
            diff = neigh_pd - pd_model

            st.markdown("### 🔀 Согласованность моделей")

            if abs(diff) < 0.1:
                st.success("✅ Скоринг и KNN согласны")
            elif diff > 0:
                st.warning("⚠️ KNN оценивает риск выше, чем скоринг")
            else:
                st.info("ℹ️ Скоринг строже, чем KNN")
 
        with st.expander("🧾 Лог решения"):
            st.json({
                "K": k_user,
                "neighbors": neighbors_count,
                "group_pd": round(neigh_pd, 3),
                "confidence": round(confidence, 2),
                "decision": decision
            })

        with st.expander("ℹ️ Как работает рекомендация"):
            st.markdown("""
            **Метод:** K-ближайших соседей (KNN)  
            **Смысл:**  
            Клиент сравнивается с исторически похожими клиентами,
            после чего анализируется их фактическое поведение.
            """)

        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📄 Обучающие данные")
    st.dataframe(df.head(20), width="stretch")
    st.subheader("🎯 Распределение дефолтов")
    st.bar_chart(df["loan_status"].value_counts(normalize=True))

    st.subheader("📊 Распределения числовых признаков")

    num_cols = (
        df.select_dtypes(exclude="object")
          .columns
          .drop("loan_status")
    )

    df[num_cols].hist(
        bins=30,
        figsize=(14, 10)
    )

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

 
    st.subheader("🔗 Корреляционная матрица")

    corr = df[num_cols.tolist() + ["loan_status"]].corr()

    fig_corr, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=False,
        ax=ax
    )
    plt.title("Correlation Matrix")
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    st.markdown("## 📅 Сезонный анализ кредитного риска")

    st.markdown("""
    Анализ показывает, как меняется вероятность дефолта клиентов
    в зависимости от месяца и квартала выдачи кредита.
    Это помогает банку понимать периоды повышенного риска.
    """)

    col1, col2 = st.columns(2) 
    with col1:
        st.markdown("### 📈 Дефолтность по месяцам")

        month_pd = df.groupby("month")["loan_status"].mean()

        fig, ax = plt.subplots(figsize=(5,4))
        month_pd.plot(marker="o", linewidth=3, ax=ax)

        ax.set_title("Средний уровень дефолта по месяцам", fontsize=12)
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Доля дефолтов")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        worst_month = month_pd.idxmax()
        best_month = month_pd.idxmin()

        st.info(f"🔴 Самый рискованный месяц: **{worst_month}**")
        st.success(f"🟢 Самый безопасный месяц: **{best_month}**")
 
    with col2:
        st.markdown("### 🏦 Дефолтность по кварталам")

        quarter_pd = df.groupby("quarter")["loan_status"].mean()

        fig2, ax2 = plt.subplots(figsize=(5,4))
        quarter_pd.plot(kind="bar", ax=ax2)

        ax2.set_title("Средний уровень дефолта по кварталам", fontsize=12)
        ax2.set_xlabel("Квартал")
        ax2.set_ylabel("Доля дефолтов")
        ax2.grid(axis="y", alpha=0.3)

        st.pyplot(fig2)

        worst_q = quarter_pd.idxmax()
        best_q = quarter_pd.idxmin()

        st.warning(f"📉 Самый рискованный квартал: **{worst_q}**")
        st.success(f"📈 Самый стабильный квартал: **{best_q}**")


    st.markdown("""
    💡 **Вывод:**  
    Сезонный анализ позволяет банку корректировать кредитную политику,
    процентные ставки и лимиты в периоды повышенного риска.
    """)



 
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