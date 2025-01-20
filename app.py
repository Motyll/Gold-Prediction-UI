import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
from data.data_loader import DataLoader
from data.data_fetch import GoldDataFetcher
from models.model import Model
from models.predictor import Predictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from pytz import UTC
import streamlit as st

st.sidebar.header("Ustawienia")

model_options = ["GradientBoosting", "RandomForest", "ExtraTrees"] #, "CatBoost"]
selected_model = st.sidebar.selectbox("Wybierz model:", model_options)

timeframe_options = {
    "2 lata": 730,
    "1 rok": 365,
    "6 miesięcy": 180,
    "3 miesiące": 90
}
selected_timeframe = st.sidebar.selectbox("Wybierz przedział czasowy:", list(timeframe_options.keys()))
days_to_train = timeframe_options[selected_timeframe]

@st.cache_data
def get_exchange_rates():
    tickers = ["USDEUR=X", "USDPLN=X", "USDGBP=X"]
    try:
        rates = yf.download(tickers, period="1d")["Close"].iloc[-1]
        return {
            "EUR": rates["USDEUR=X"],
            "PLN": rates["USDPLN=X"],
            "GBP": rates["USDGBP=X"],
            "USD": 1.0
        }
    except Exception as e:
        st.error(f"Nie udało się pobrać danych kursów wymiany walut. Błąd: {e}")
        return {
            "EUR": None,
            "PLN": None,
            "GBP": None,
            "USD": 1.0
        }

exchange_rates = get_exchange_rates()
selected_currency = st.sidebar.selectbox("Wybierz walutę:", options=exchange_rates.keys())
exchange_rate = exchange_rates[selected_currency]
st.sidebar.write(f"Aktualny kurs wymiany: 1 USD = {exchange_rate:.2f} {selected_currency}")

def fetch_gold_data():
    data_fetcher = GoldDataFetcher(output_file="gold_hourly_data_transformed.csv")
    gold_data = data_fetcher.fetch_data()
    return gold_data

def retrain_model():
    filepath = 'gold_hourly_data_transformed.csv'
    loader = DataLoader(filepath=filepath)
    loader.load_data()

    start_date = datetime.now(UTC) - timedelta(days=days_to_train)
    loader.data.index = pd.to_datetime(loader.data.index).tz_convert(UTC)
    loader.data = loader.data[loader.data.index >= start_date]

    X, y = loader.prepare_features()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.values)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # dane testowe to ostatnie 20% danych
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    model = Model(model_type=selected_model)
    model.train(X_train, y_train)

    mse_test = model.evaluate(X_test, y_test)

    return model, scaler_X, scaler_y, X, y, mse_test

if st.sidebar.button("Uruchom uczenie modelu"):
    with st.spinner("Uczenie modelu, proszę czekać..."):
        model, scaler_X, scaler_y, X, y, metrics = retrain_model()

        st.success("Model został ponownie wytrenowany!")
        st.subheader(f"Predykowane ceny złota na kolejny dzień:")

        gold = yf.Ticker("GC=F")
        gold_price_usd = gold.history(period="1d")["Close"][0]
        st.write(
            f"Dzisiejsza cena złota: {gold_price_usd:.2f} USD / {(gold_price_usd * exchange_rate):.2f} {selected_currency}"
        )

        predictor = Predictor(model)
        last_week_data = X[-7 * 8:]
        last_week_scaled = scaler_X.transform(last_week_data.values)
        last_known_features = last_week_scaled[-1].reshape(1, -1)
        current_day = datetime.now()

        day_of_prediction = current_day.strftime("%A")
        day_of_prediction = "Monday" #usunac
        if day_of_prediction in ["Saturday", "Sunday"]:
            st.warning(f"Prognoza cen złota dla {day_of_prediction} (weekend) nie jest możliwa. Giełda jest zamknięta.")
        else:
            predicted_prices_scaled = predictor.predict_next_days(last_known_features[0], current_day)
            predicted_prices_usd = [scaler_y.inverse_transform([[price]])[0, 0] for price in predicted_prices_scaled[0]]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Tabela prognoz:**")
                hours = list(range(9, 17))
                data_display = pd.DataFrame({
                    "Godzina": [f"{hour}:00" for hour in hours],
                    "Cena w USD": [f"{price:.2f}" for price in predicted_prices_usd],
                    f"Cena w {selected_currency}": [f"{price * exchange_rate:.2f}" for price in predicted_prices_usd]
                })
                st.dataframe(data_display)

            with col2:
                st.write("**Wykres cen:**")
                plt.figure(figsize=(5, 4))
                plt.plot([f"{hour}:00" for hour in hours], predicted_prices_usd, marker='o', color='blue', label="Cena w USD")
                plt.axhline(y=gold_price_usd, color='red', linestyle='--', label=f"Dzisiejsza cena ({gold_price_usd:.2f} USD)")
                plt.xticks(rotation=45)
                plt.xlabel("Godzina")
                plt.ylabel("Cena Złota (USD)")
                plt.title(f"Przewidywane ceny na dzień: {day_of_prediction}")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)