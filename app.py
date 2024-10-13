from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from functions.functions import load_data, extract_features_from_custom, predict_score, generate_shap_image

# Charger les données
df, customer_ids = load_data()

app = Flask(__name__)

@app.route('/')
def welcome():
    # Retourne la liste des IDs clients en JSON pour potentiellement l'utiliser côté client
    return jsonify({"customer_ids": customer_ids})

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer l'ID du client à partir de la requête JSON envoyée par Streamlit
    selected_id = int(request.json.get('customer_id'))
    
    # Extraire les caractéristiques du client
    customer_data = extract_features_from_custom(df, selected_id)
    
    # Réaliser la prédiction
    decision, prediction_success, prediction_failure = predict_score(customer_data)
    
    # Retourne les résultats de la prédiction sous forme de JSON
    return jsonify({
        "decision": decision,
        "prediction_success": prediction_success,
        "prediction_failure": prediction_failure,
        "customer_id": selected_id
    })

@app.route('/result', methods=['GET'])
def show_prediction():
    # Récupère les informations de la requête pour les renvoyer en JSON
    decision = request.args.get('decision')
    prediction_success = request.args.get('prediction_success')
    prediction_failure = request.args.get('prediction_failure')
    customer_id = request.args.get('customer_id')
    
    # Retourner les informations du résultat sous forme de JSON pour usage client
    return jsonify({
        "decision": decision,
        "prediction_success": prediction_success,
        "prediction_failure": prediction_failure,
        "customer_id": customer_id
    })

@app.route('/explain/<int:customer_id>', methods=['GET'])
def explain(customer_id):
    try:
        # Extraire les données du client
        customer_data_raw = extract_features_from_custom(df, customer_id)
        print("Extraction des données du client réussie")
        
        # Générer l'image SHAP et récupérer le chemin de l'image
        plot_path = generate_shap_image(customer_data_raw)
        
        # Retourner le chemin de l'image sous forme de JSON pour affichage dans Streamlit
        return jsonify({"image_url": plot_path})
    
    except Exception as e:
        print("Erreur dans la route explain:", str(e))
        return jsonify({"error": "Une erreur s'est produite lors de la génération de l'explication."}), 500

if __name__ == '__main__':
    app.run(debug=True)

