import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pandas as pd

# Sélecteur de produit
st.title("Systéme de prédiction des prix des produits agricoles")
#product_options = data['commodity'].unique().tolist()  # Obtenez les produits uniques du dataset
product_options = ['Mil', 'Sorgho', 'Maïs', 'Riz (importé)', 'Riz (local)', 'Soja','Arachide',
                   'Ble', 'gari', 'Chou', 'Igname', 'Oignons', 'Tomates']

selected_product = st.sidebar.selectbox("Choisissez le produit:", product_options)

# Charger le modèle ARIMA et le scaler spécifique au produit sélectionné
model_filename = f'arima_model_{selected_product}.pkl'  # Nommer les modèles selon les produits
#scaler_filename = f'scaler_{selected_product}.pkl'  # Nommer les scalers selon les produits

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Fonction pour effectuer les prédictions
def predict_prices(days_to_predict, product):
    # Filtrer les données pour le produit sélectionné
    product_data =selected_product #data[data['commodity'] == product]
    
    # Prédiction basée sur les données filtrées
    forecast = loaded_model.forecast(steps=days_to_predict)
    return forecast

# Input du nombre de jours à prédire
days_to_predict = st.sidebar.number_input("Entrez le nombre de jours à prédire", min_value=1, max_value=30, value=7)

# Bouton pour lancer la prédiction
if st.sidebar.button("Prédire"):
    current_date = datetime.now()
    forecast_prices = predict_prices(days_to_predict, selected_product)

    # Afficher les prédictions en chiffres
    st.subheader(f"Prédiction des prix du {selected_product} pour les {days_to_predict} prochains jours :")
    predictions_data = []

    for i, price in enumerate(forecast_prices):
        prediction_date = current_date + timedelta(days=i)
        predictions_data.append((prediction_date, price))
        st.write(f"{prediction_date.strftime('%Y-%m-%d')}: {price:.2f} fcfa/kg")

    # Graphique des prédictions
    st.subheader(f"fluctuation du {selected_product} pour les {days_to_predict} prochains jours :")
    x_dates = [date for date, price in predictions_data]
    y_prices = [price for date, price in predictions_data]

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, y_prices, marker='o', label=f'Prédictions pour {selected_product}')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.title(f'Prédictions du prix du {selected_product} pour les {days_to_predict} prochains jours')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
