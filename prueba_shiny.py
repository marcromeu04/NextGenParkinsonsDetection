# shiny_app.py
from shiny import App, ui, reactive
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from autogluon.tabular import TabularPredictor

# Cargar modelos y variables al iniciar
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


# Interfaz
app_ui = ui.page_fluid(
    ui.h2("Predicción clínica para paciente individual (stacking)"),
    ui.input_file("file", "Cargar paciente (.pkl)", accept=[".pkl"]),
    ui.input_action_button("procesar", "Procesar paciente"),
    ui.output_text_verbatim("resultado")
)

# Server
def server(input, output, session):
    @reactive.event(input.procesar)
    def resultado_final():
        file_info = input.file()
        if not file_info:
            return "Por favor, carga un archivo .pkl"

        # Leer el archivo del paciente
        paciente = pd.read_pickle(file_info[0]['datapath'])

        # Suponiendo que ya está expandido o es 1 fila con todas las features
        X_cues = paciente[selected_vars_cues]
        X_psd = paciente[selected_vars_psd]
        X_esta = paciente[selected_vars_esta]

        # Procesar boss (aquí deberías aplicar el modelo BOSS a cada señal si necesario)
        X_boss = paciente[selected_vars_boss]

        # Concatenar probabilidades simuladas (aquí normalmente usarías modelos predictivos previos)
        # Simulación: en vez de predict_proba() usamos directamente X_* como si fueran predicciones
        X_stack = np.hstack([X_cues.values, X_psd.values, X_esta.values, X_boss.values])

        # Predecir
        y_prob = meta_model.predict_proba(X_stack)
        y_pred = np.argmax(y_prob, axis=1)[0]

        clases = ["Healthy", "Parkinson", "Other movement disorder"]
        return f"Predicción final: {clases[y_pred]} (probabilidades: {np.round(y_prob[0], 3).tolist()})"

    output.resultado = ui.render_text(resultado_final)

# Lanzar app
app = App(app_ui, server)