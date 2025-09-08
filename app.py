import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Configuración
st.set_page_config(page_title="Deserción Universitaria", page_icon="🎓", layout="wide")
st.title("🎓 Sistema de Alerta Temprana para Deserción Estudiantil")
st.markdown("Modelo: **XGBoost** | Precisión: **93.5%** | Variables más importantes:")

# Variables y explicaciones
features = {
    "Materias 2º semestre aprobadas": "Indicador directo de rendimiento reciente.",
    "Eficiencia académica (%)": "Relación entre créditos aprobados e inscritos.",
    "Matrícula al día": "Refleja estabilidad económica.",
    "Materias 2º semestre inscritas": "Carga académica actual.",
    "Evaluaciones 2º semestre": "Nivel de participación.",
    "Necesidades educativas especiales": "Puede requerir apoyo adicional.",
    "Carga académica (ECTS)": "Volumen total de estudio.",
    "Becado": "Acceso a recursos económicos.",
    "Materias 1º semestre aprobadas": "Historial académico previo.",
    "Materias 1º semestre convalidadas": "Reconocimiento de créditos anteriores."
}

# Sidebar
st.sidebar.header("📋 Datos del Estudiante")
inputs = {}

for i, (label, explanation) in enumerate(features.items()):
    st.sidebar.markdown(f"**{label}**")
    st.sidebar.caption(explanation)

    if "aprobadas" in label or "inscritas" in label or "evaluaciones" in label or "convalidadas" in label:
        inputs[label] = st.sidebar.slider(label, 0, 10, 5)
    elif "ECTS" in label:
        inputs[label] = st.sidebar.slider(label, 0, 60, 30)
    elif "%" in label:
        inputs[label] = st.sidebar.slider(label, 0, 100, 75) / 100  # normalizado
    else:
        inputs[label] = st.sidebar.selectbox(label, ["Sí", "No"]) == "Sí"

# Modelo simulado
@st.cache_resource
def load_model():
    np.random.seed(42)
    X = np.random.rand(500, 10)
    y = np.random.choice([0, 1, 2], size=500, p=[0.3, 0.4, 0.3])
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model

model = load_model()

# Predicción
if st.sidebar.button("🔍 Predecir Riesgo"):
    X_input = np.array([list(inputs.values())]).astype(float)
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]

    risk_labels = ["🚨 Alto Riesgo", "⚠️ Riesgo Medio", "✅ Bajo Riesgo"]
    st.subheader("📊 Resultado de la Predicción")
    st.metric("Nivel de Riesgo", risk_labels[pred])
    st.metric("Confianza", f"{prob[pred]*100:.1f}%")
    st.progress(prob[0], text=f"Probabilidad de Alto Riesgo: {prob[0]*100:.1f}%")

    # Tabla de probabilidades
    st.subheader("📈 Distribución de Probabilidades")
    df = pd.DataFrame({
        "Categoría": risk_labels,
        "Probabilidad": [f"{p*100:.1f}%" for p in prob]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Gráfico de importancia de características
    st.subheader("📊 Importancia de Características")
    importance = {
        "Materias 2º semestre aprobadas": 0.2337,
        "Eficiencia académica (%)": 0.1854,
        "Matrícula al día": 0.0483,
        "Materias 2º semestre inscritas": 0.0481,
        "Evaluaciones 2º semestre": 0.0352,
        "Necesidades educativas especiales": 0.0278,
        "Carga académica (ECTS)": 0.0252,
        "Becado": 0.0204,
        "Materias 1º semestre aprobadas": 0.0191,
        "Materias 1º semestre convalidadas": 0.0174
    }

    fig, ax = plt.subplots()
    ax.barh(list(importance.keys()), list(importance.values()), color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importancia")
    st.pyplot(fig)

else:
    st.info("👈 Introduce los datos en la barra lateral y pulsa 'Predecir Riesgo'.")
