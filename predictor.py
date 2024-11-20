import pandas as pd
from datetime import datetime, timedelta

class Predictor:
    def __init__(self, model):
        self.model = model

    def predict_next_days(self, last_known_features, start_day, hours=8):
        predictions = []
        day_count = 0

        while day_count < 5:
            start_day += timedelta(days=1)
            if start_day.weekday() >= 5:  # Pomijanie weekendu
                continue

            daily_predictions = []
            current_features = last_known_features.reshape(1, -1)  # Przekazujemy jako jednowymiarową tablicę

            for hour in range(hours):  # 8 godzin (9:00 do 17:00)
                predicted_price = self.model.predict(current_features)[0]
                daily_predictions.append(predicted_price)

                previous_price = predicted_price
                two_hour_avg = (previous_price + current_features[0][0]) / 2
                four_hour_avg = (previous_price + sum(current_features[0][:3])) / 4

                current_features = pd.DataFrame([[previous_price, two_hour_avg, four_hour_avg]]).values

            predictions.append(daily_predictions)
            last_known_features = current_features  # Aktualizujemy cechy na podstawie końcowej godziny

            day_count += 1

        return predictions
