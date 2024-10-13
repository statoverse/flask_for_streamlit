from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from functions.functions import load_data, extract_features_from_custom, predict_score, generate_shap_image, generate_feature_distributions

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
        
        # Générer le graphique SHAP et l'enregistrer dans un buffer mémoire
        fig, ax = plt.subplots(figsize=(27, 10))
        generate_shap_image(customer_data_raw, ax)  # Remplacez par votre fonction de génération SHAP
        plt.tight_layout()
        
        # Sauvegarder l'image dans un buffer en mémoire
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)  # Ferme la figure pour libérer de la mémoire
        buf.seek(0)
        
        # Encoder l'image en base64
        image_base64 = base64.b64encode(buf.getbuffer()).decode("ascii")
        
        # Créer la balise HTML pour l'image
        image_html = f"<img src='data:image/png;base64,{image_base64}' style='max-width:100%; height:auto;'/>"
        
        # Retourner la balise HTML directement
        return Response(image_html, mimetype='text/html')
    except Exception as e:
        print("Erreur dans la route explain:", str(e))
        return Response("<p>Une erreur s'est produite lors de la génération de l'explication.</p>", mimetype='text/html', status=500)

@app.route('/distributions/<int:customer_id>', methods=['GET'])
def distributions(customer_id):
    import json
    import plotly
    try:
        fig = generate_feature_distributions(df, customer_id)
        fig_json = json.loads(plotly.io.to_json(fig))
        return jsonify(fig_json)
    except Exception as e:
        print("Erreur dans la génération des distributions:", str(e))
        return jsonify({"error": "Une erreur s'est produite"}), 500



if __name__ == '__main__':
    app.run(debug=True)

