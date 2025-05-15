
from shiny import App, ui, render
import pandas as pd
import joblib
import shap

import matplotlib.pyplot as plt
import numpy as np
from autogluon.tabular import TabularPredictor
import plotly.express as px

# Cargar paciente
with open("paciente_347.pkl", "rb") as f:
    paciente = joblib.load(f)

# Cargar bases de datos
with open("boss_features.pkl", "rb") as f:
    base_datos_boss = joblib.load(f)
with open("estadisticos_features.pkl", "rb") as f:
    base_datos_esta = joblib.load(f)
base_datos_psd = joblib.load("psd_features.pkl")
with open("DatosCompletos.pkl", "rb") as f:
    base_datos = joblib.load(f)

# Cargar modelos y variables
meta_model = joblib.load("meta_model_xgb.pkl")
selected_vars_cues = joblib.load("selected_vars_cues.pkl")
selected_vars_boss = joblib.load("selected_vars_boss.pkl")
selected_vars_esta = joblib.load("selected_vars_esta.pkl")
selected_vars_psd = joblib.load("selected_vars_psd.pkl")
boss_models = joblib.load("boss_models.pkl")

# Cargar AutoML
predictor_cues = TabularPredictor.load("AutogluonModels/ag-20250508_072135")
predictor_boss = TabularPredictor.load("AutogluonModels/ag-20250508_093716")
predictor_esta = TabularPredictor.load("AutogluonModels/ag-20250508_094129")
predictor_psd = TabularPredictor.load("AutogluonModels/ag-20250508_101709")

# Filtrar variables
X_cues = pd.DataFrame([paciente])[selected_vars_cues]
X_boss = pd.DataFrame([base_datos_boss.loc[347]])[selected_vars_boss]
X_esta = pd.DataFrame([base_datos_esta.loc[347]])[selected_vars_esta]
X_psd = pd.DataFrame([base_datos_psd.loc[347]])[selected_vars_psd]

# Predicciones
prob_cues = predictor_cues.predict_proba(X_cues).values[0]
prob_boss = predictor_boss.predict_proba(X_boss).values[0]
prob_esta = predictor_esta.predict_proba(X_esta).values[0]
prob_psd = predictor_psd.predict_proba(X_psd).values[0]

# Meta-modelo stacking
X_stack = np.hstack([prob_psd, prob_cues, prob_esta, prob_boss])
prob_meta = meta_model.predict_proba(X_stack.reshape(1, -1))[0]
final_pred = np.argmax(prob_meta)

# SHAP
explainer = shap.Explainer(predictor_cues.predict_proba, X_cues)
shap_values = explainer(X_cues)

# UI
app_ui = ui.page_fluid(
    ui.h2("Predicción del paciente 347"),
    ui.output_text("predicciones"),
    ui.output_plot("shap_plot"),
    ui.output_table("shap_top")
)

# Server
def server(input, output, session):
    @output
    @render.text
    def predicciones():
        etiquetas = ["Healthy", "Parkinson", "Other"]
        return (
            f"Modelo CUES: {etiquetas[np.argmax(prob_cues)]}\n"
            f"Modelo BOSS: {etiquetas[np.argmax(prob_boss)]}\n"
            f"Modelo ESTA: {etiquetas[np.argmax(prob_esta)]}\n"
            f"Modelo PSD: {etiquetas[np.argmax(prob_psd)]}\n"
            f"Predicción Final (Meta-modelo): {etiquetas[final_pred]}"
        )

    @output
    @render.plot
    def shap_plot():
        shap.plots.bar(shap_values[0], show=False)
        plt.title("Importancia de variables (CUES)")
        return plt.gcf()

    @output
    @render.table
    def shap_top():
        top_features = pd.DataFrame({
            "feature": X_cues.columns,
            "SHAP": shap_values.values[0]
        })
        top_features["abs_SHAP"] = np.abs(top_features["SHAP"])
        return top_features.sort_values("abs_SHAP", ascending=False).head(10)[["feature", "SHAP"]]

app = App(app_ui, server)
