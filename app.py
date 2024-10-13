import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.io as pio
import json
import time
from streamlit_option_menu import option_menu

# Charger les données clients
data_path = 'data/customers.csv'
df = pd.read_csv(data_path)
customer_ids = df['SK_ID_CURR'].astype(str).tolist()

# URL de base pour l'API déployée sur Heroku
base_url = "https://backendflaskforscoring-12eba9fb5ac8.herokuapp.com"

# Configuration de la barre latérale avec une liste déroulante pour sélectionner l'ID du client
with st.sidebar:
    st.write("### Sélectionnez un Client ID")
    selected_customer_id = st.selectbox("Client ID", customer_ids)
    num_features = st.slider("Nombre de features SHAP à afficher", min_value=1, max_value=20, value=5)
    decision_threshold = st.slider("Seuil de décision", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Navigation entre les trois sous-panneaux
selected_panel = option_menu(
    menu_title=None,
    options=["Résultat Prêt", "Graphique SHAP", "Distributions"],
    icons=["check-circle", "chart-bar", "chart-line"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Vérification de la sélection et récupération des données de prédiction
if selected_customer_id:
    response = requests.post(f"{base_url}/predict", json={"customer_id": selected_customer_id})
    
    if response.ok:
        data = response.json()
        prediction_success = data.get("prediction_success")
        prediction_failure = data.get("prediction_failure")
        decision = data.get("decision")
        
        # Page 1 : Jauge et Résultat de Prêt
        if selected_panel == "Résultat Prêt":
            st.write("## Résultat de l'accord de prêt")
            decision_finale = "Accordé" if prediction_failure <= decision_threshold else "Refusé"
            st.write(f"**Décision finale à un seuil de {decision_threshold}:** {decision_finale}")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_failure,
                title={"text": "Probabilité de défaut de paiement"},
                gauge={'axis': {'range': [0, 1]},
                       'steps': [
                           {'range': [0, decision_threshold], 'color': "lightgreen"},
                           {'range': [decision_threshold, 1], 'color': "red"}]},
            ))
            st.plotly_chart(gauge)

        # Page 2 : Graphique SHAP (adapté pour base64)
        elif selected_panel == "Graphique SHAP":
            st.write("## Explication du modèle : SHAP")
            shap_response = requests.get(f"{base_url}/explain/{selected_customer_id}")
            if shap_response.ok:
                shap_data = shap_response.json()
                image_base64 = shap_data.get("image_base64")
                if image_base64:
                    # Générer l'élément HTML pour afficher l'image en base64
                    image_html = f"<img src='data:image/png;base64,{image_base64}'/>"
                    st.markdown(image_html, unsafe_allow_html=True)
                else:
                    st.error("Erreur: L'image SHAP n'a pas pu être chargée.")
            else:
                st.error("Erreur lors de la récupération du graphique SHAP.")

        # Page 3 : Distributions des Features
        elif selected_panel == "Distributions":
            st.write("## Positionnement du client par rapport aux autres clients")
            dist_response = requests.get(f"{base_url}/distributions/{selected_customer_id}")
            if dist_response.ok:
                fig_data = dist_response.json()
                fig = pio.from_json(json.dumps(fig_data))
                st.plotly_chart(fig)
            else:
                st.error("Erreur lors de la récupération des distributions des features.")
    else:
        st.error("Erreur lors de la récupération des données de prédiction. Veuillez vérifier l'ID client et réessayer.")
else:
    st.info("Veuillez sélectionner un Client ID dans la barre latérale pour commencer.")
