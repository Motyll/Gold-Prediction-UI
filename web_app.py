import streamlit as st
from datetime import datetime, timedelta
from data.data_loader import DataLoader
from model import Model
from predictor import Predictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt

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

st.header("Przewidywanie cen złota na nadchodzący tydzień")
last_week_data = X[-7*8:]
last_week_scaled = scaler_X.transform(last_week_data.values)
predictor = Predictor(model)
last_known_features = last_week_scaled[-1].reshape(1, -1)
current_day = datetime.now()

predicted_prices_scaled = predictor.predict_next_days(last_known_features[0], current_day)
predicted_prices = [
    [scaler_y.inverse_transform([[price]])[0, 0] for price in day] for day in predicted_prices_scaled
]

gold = yf.Ticker("GC=F")
gold_price = gold.history(period="1d")['Close'][0]
st.write(f"Dzisiejsza cena złota: {gold_price:.2f} USD")

days_of_week = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]
all_dates = []
all_prices = []

st.subheader("Predykcje cen złota na nadchodzący tydzień")
for day_idx, daily_prices in enumerate(predicted_prices):
    day_name_index = (current_day.weekday() + day_idx + 1) % 7
    day_name = days_of_week[day_name_index]

    if day_name_index >= 5:  # Skip weekends
        st.write(f"**{day_name}: Rynek zamknięty (weekend)**")
        continue

    st.write(f"### {day_name} (rynek otwarty 9:00–17:00):")
    hourly_prices = {}
    for hour, price in enumerate(daily_prices, start=9):
        hourly_prices[f"{hour}:00"] = price
        all_dates.append(f"{day_name} {hour}:00")
        all_prices.append(price)

    st.write(hourly_prices)

st.subheader("Predykowane ceny złota na nadchodzący tydzień")
plt.figure(figsize=(12, 6))
plt.plot(all_dates, all_prices, marker='o', color='blue', label="Predicted Price")
plt.axhline(y=gold_price, color='red', linestyle='--', label=f"Today's Price: {gold_price:.2f} USD")
plt.xticks(rotation=45, ha='right')
plt.xlabel("Data i Godzina")
plt.ylabel("Celna Złota (USD)")
plt.title("Przewidywane ceny złota na nadchodzący tydzień")
plt.legend()
plt.tight_layout()
plt.grid(True)

st.pyplot(plt)
