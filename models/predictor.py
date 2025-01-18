import pandas as pd
from datetime import timedelta
import numpy as np


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict_next_days(self, last_known_features, start_day, hours=8):
        predictions = []
        day_count = 0

        while day_count < 5:
            start_day += timedelta(days=1)

            # Pomijamy weekendy
            if start_day.weekday() >= 5:
                continue

            daily_predictions = []
            current_features = last_known_features.reshape(1, -1)

            for hour in range(hours):
                predicted_price = self.model.predict(current_features)[0]
                daily_predictions.append(predicted_price)

                previous_price = predicted_price
                two_hour_avg = (previous_price + current_features[0][0]) / 2
                four_hour_avg = (previous_price + sum(current_features[0][:3])) / 4

                twelve_hour_avg = (previous_price + sum(current_features[0][:11])) / 12 if len(current_features[0]) >= 12 else previous_price
                twenty_four_hour_avg = (previous_price + sum(current_features[0][:23])) / 24 if len(current_features[0]) >= 24 else previous_price

                volatility = np.std([previous_price, current_features[0][0]])
                price_change = ((previous_price - current_features[0][0]) / current_features[0][0]) * 100

                current_features = pd.DataFrame([[
                    previous_price,  # Poprzednia cena
                    two_hour_avg,    # Średnia z 2 godzin
                    four_hour_avg,   # Średnia z 4 godzin
                    twelve_hour_avg, # Średnia z 12 godzin
                    twenty_four_hour_avg, # Średnia z 24 godzin
                    volatility,      # Zmienność
                    price_change,    # Zmiana procentowa
                    start_day.weekday(),  # Dzień tygodnia (0-6)
                    hour,            # Godzina dnia (0-23)
                    current_features[0][-1]  # Wolumen
                ]]).values

            predictions.append(daily_predictions)
            last_known_features = current_features[0]
            day_count += 1

        return predictions