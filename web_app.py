import streamlit as st
from datetime import datetime, timedelta
from data.data_loader import DataLoader
from model import Model
from predictor import Predictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import time

st.sidebar.header("Ustawienia")
st.sidebar.subheader("Kurs wymiany walut")


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


exchange_rates = get_exchange_rates()
selected_currency = st.sidebar.selectbox("Wybierz walutę:", options=exchange_rates.keys())
exchange_rate = exchange_rates[selected_currency]
st.sidebar.write(f"Aktualny kurs wymiany: 1 USD = {exchange_rate:.2f} {selected_currency}")

if st.sidebar.button("Aktualizuj dane"):
    with st.spinner("Aktualizowanie danych, proszę czekać..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)
        st.success("Dane zostały zaktualizowane!")


def retrain_model():
    filepath = 'data/gold_hourly_data_transformed.csv'
    loader = DataLoader(filepath=filepath)
    loader.load_data()
    X, y = loader.prepare_features()

    last_three_months = X.index > (X.index[-1] - timedelta(days=90))
    X_recent = X[last_three_months]
    y_recent = y[last_three_months]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_recent.values)
    y_scaled = scaler_y.fit_transform(y_recent.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    model = Model()
    model.train(X_train, y_train)
    mse_test = model.evaluate(X_test, y_test)

    return model, scaler_X, scaler_y, X, y, mse_test

if st.sidebar.button("Uruchom uczenie modelu"):
    with st.spinner("Uczenie modelu, proszę czekać..."):
        model, scaler_X, scaler_y, X, y, mse_test = retrain_model()
        st.success("Model został ponownie wytrenowany!")
        st.title("Predykcja cen złota na następny dzień")
        st.write(f"Średni błąd kwadratowy (MSE) na danych testowych: {mse_test:.4f}")

        gold = yf.Ticker("GC=F")
        gold_price_usd = gold.history(period="1d")['Close'][0]
        st.write(
            f"Dzisiejsza cena złota: {gold_price_usd:.2f} USD / {(gold_price_usd * exchange_rate):.2f} {selected_currency}")

        predictor = Predictor(model)
        last_week_data = X[-7 * 8:]
        last_week_scaled = scaler_X.transform(last_week_data.values)
        last_known_features = last_week_scaled[-1].reshape(1, -1)
        current_day = datetime.now()

        predicted_prices_scaled = predictor.predict_next_days(last_known_features[0], current_day)

        predicted_prices_usd = [scaler_y.inverse_transform([[price]])[0, 0] for price in predicted_prices_scaled[0]]

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
