# NextGenParkinsonsDetection

## Project Description

This repository hosts the "NextGenParkinsonsDetection" project, aimed at developing an advanced system for detecting Parkinson's disease and its differential diagnosis using the PADS dataset from the paper
https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/.

This project is a collaborative effort by a group of students as part of the **Project III** course in the Degree in Data Science at the Universitat Politècnica de València (UPV).

## Repository Content

The project encompasses several key areas:

* **Exploratory Data Analysis (EDA)**: Initial processing and in-depth analysis of the dataset.
* **Predictive Modeling**:
    * Models based on questionnaire data (clinical and demographic information).
    * Models based on time series data extracted from wearable sensors (accelerometers, gyroscopes).
    * Exploration of various feature extraction techniques for time series, including:
        * Descriptive statistics
        * Power Spectral Density (PSD) based features
        * BOSS (Bag-of-SFA-Symbols) transform
    * Training and evaluation of diverse classification algorithms such as:
        * Random Forest
        * XGBoost
        * AutoML-generated models (AutoGluon)
    * Implementation of a meta-model (stacking) that combines predictions from individual models to enhance overall accuracy.
    * Estraction of feature importance using Shapely Values 
* **Interactive Application**: A user-friendly application developed with Shiny to perform predictions on new patient data and visualize feature importance.

## Data

The primary dataset used for training and evaluating the models is located in the `DatosFinal.pkl` file.

As seen in the Jupyter notebooks (`mockupVersionFinal.ipynb`, `shiny_app.ipynb`), the project also involves loading and processing other data files, including:
* `DatosCompletos.pkl`
* `paciente_347.pkl` (example patient data)
* `boss_features.pkl`
* `estadisticos_features.pkl`
* `psd_features.pkl`

The classification objective is to categorize subjects into one of the following groups:
* **HC**: Healthy Control
* **PD**: Parkinson's Disease
* **DD**: Differential Diagnosis (other movement disorders)

## Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/marcromeu04/NextGenParkinsonsDetection.git](https://github.com/marcromeu04/NextGenParkinsonsDetection.git)
    cd NextGenParkinsonsDetection
    ```
2.  **Set up the environment** (creating a virtual environment is highly recommended):
    ```bash
    pip install pandas numpy scikit-learn xgboost autogluon shap pyts joblib shiny matplotlib plotly 
    ```
3.  **Run analyses and models:**
    Explore the Jupyter notebooks (e.g., `mockupVersionFinal.ipynb`) for detailed analysis and model training steps.

4.  **Launch the Shiny application:**
   Navigate to the directory containing the Shiny app script (e.g., `shiny_app.py`) and run:
    ```bash
    shiny run shiny_app.py
    ```
## Collaborators

*This project was a collaborative effort.
*Elsa Guerra Elena
*Maru Mendez Ortuondo
*Carla Sala Moncho
*Carla Salavert Martí
*Elena Adán Lledó
*Marc Romeu Ferrás


---
