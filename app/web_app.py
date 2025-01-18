import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime, timedelta
from data.data_loader import DataLoader
from models.model import Model
from models.predictor import Predictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from pytz import UTC

# Konfiguracja bocznego menu
st.sidebar.header("Ustawienia")
st.sidebar.subheader("Kurs wymiany walut")


# Funkcja do pobierania kursów wymiany walut
@st.cache_data
def get_exchange_rates():
    tickers = ["USDEUR=X", "USDPLN=X", "USDGBP=X"]
    rates = yf.download(tickers, period="1d")['Close'].iloc[-1]
    return {
        "EUR": rates["USDEUR=X"],
        "PLN": rates["USDPLN=X"],
        "GBP": rates["USDGBP=X"],
        "USD": 1.0
    }


# Pobierz kursy wymiany walut
exchange_rates = get_exchange_rates()
selected_currency = st.sidebar.selectbox("Wybierz walutę:", options=exchange_rates.keys())
exchange_rate = exchange_rates[selected_currency]
st.sidebar.write(f"Aktualny kurs wymiany: 1 USD = {exchange_rate:.2f} {selected_currency}")


# Funkcja do ponownego treningu modelu na ostatnich 3 miesiącach
def retrain_model():
    filepath = 'data/gold_hourly_data_transformed.csv'
    loader = DataLoader(filepath=filepath)
    loader.load_data()

    # Filtrowanie danych do ostatnich 3 miesięcy
    three_months_ago = datetime.now(UTC) - timedelta(days=90)  # Konwersja do UTC
    loader.data.index = pd.to_datetime(loader.data.index).tz_convert(UTC)  # Konwersja indeksu do UTC
    loader.data = loader.data[loader.data.index >= three_months_ago]

    # Przygotowanie cech i celu
    X, y = loader.prepare_features()

    # Skalowanie danych
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.values)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    # Trening modelu
    model = Model()
    model.train(X_train, y_train)

    # Ewaluacja modelu
    mse_test = model.evaluate(X_test, y_test)

    return model, scaler_X, scaler_y, X, y, mse_test


# Obsługa przycisku do treningu modelu
if st.sidebar.button("Uruchom uczenie modelu"):
    with st.spinner("Uczenie modelu, proszę czekać..."):
        model, scaler_X, scaler_y, X, y, mse_test = retrain_model()
        st.success("Model został ponownie wytrenowany!")
        st.title("Predykcja cen złota na następny dzień")
        st.write(f"Średni błąd kwadratowy (MSE) na danych testowych: {mse_test:.4f}")

        # Pobranie bieżącej ceny złota
        gold = yf.Ticker("GC=F")
        gold_price_usd = gold.history(period="1d")['Close'][0]
        st.write(
            f"Dzisiejsza cena złota: {gold_price_usd:.2f} USD / {(gold_price_usd * exchange_rate):.2f} {selected_currency}"
        )

        # Przewidywanie przyszłych cen
        predictor = Predictor(model)
        last_week_data = X[-7 * 8:]
        last_week_scaled = scaler_X.transform(last_week_data.values)
        last_known_features = last_week_scaled[-1].reshape(1, -1)
        current_day = datetime.now()

        predicted_prices_scaled = predictor.predict_next_days(last_known_features[0], current_day)

        predicted_prices_usd = [scaler_y.inverse_transform([[price]])[0, 0] for price in predicted_prices_scaled[0]]

        # Wyświetlanie tabeli z prognozami
        hours = list(range(9, 17))
        data_display = pd.DataFrame({
            "Godzina": [f"{hour}:00" for hour in hours],
            "Cena w USD": [f"{price:.2f}" for price in predicted_prices_usd],
            f"Cena w {selected_currency}": [f"{price * exchange_rate:.2f}" for price in predicted_prices_usd]
        })

        st.write(data_display.style.set_table_styles([
            {
                'selector': 'thead th',
                'props': [('background-color', '#4CAF50'), ('color', 'white'), ('text-align', 'center'),
                          ('font-size', '17px')]
            },
            {
                'selector': 'tbody td',
                'props': [('text-align', 'center'), ('font-size', '17px')]
            },
        ]).set_properties(**{'text-align': 'center'}))

        # Wykresy prognoz
        st.subheader("Predykowane ceny złota w USD")
        plt.figure(figsize=(10, 5))
        plt.plot([f"{hour}:00" for hour in hours], predicted_prices_usd, marker='o', color='blue', label="Cena w USD")
        plt.axhline(y=gold_price_usd, color='red', linestyle='--', label=f"Dzisiejsza cena ({gold_price_usd:.2f} USD)")
        plt.xticks(rotation=45)
        plt.xlabel("Godzina")
        plt.ylabel("Cena Złota (USD)")
        plt.title("Przewidywane ceny złota na następny dzień")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        st.subheader(f"Predykowane ceny złota w {selected_currency}")
        plt.figure(figsize=(10, 5))
        predicted_prices_currency = [price * exchange_rate for price in predicted_prices_usd]
        plt.plot([f"{hour}:00" for hour in hours], predicted_prices_currency, marker='o', color='green',
                 label=f"Cena w {selected_currency}")
        plt.axhline(y=gold_price_usd * exchange_rate, color='red', linestyle='--',
                    label=f"Dzisiejsza cena ({gold_price_usd * exchange_rate:.2f} {selected_currency})")
        plt.xticks(rotation=45)
        plt.xlabel("Godzina")
        plt.ylabel(f"Cena Złota ({selected_currency})")
        plt.title(f"Przewidywane ceny złota na następny dzień w {selected_currency}")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)