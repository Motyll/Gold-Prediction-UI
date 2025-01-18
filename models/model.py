from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Model:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.01, max_depth=8, random_state=42
        )

    def train(self, X_train, y_train):
        """Trenuje model na danych treningowych."""
        print("Rozpoczynanie treningu modelu...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Ewaluacja modelu na danych testowych."""
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }

        print("Metryki ewaluacji modelu:")
        for key, value in metrics.items():
            print(f"- {key}: {value:.4f}")

        return metrics

    def predict(self, X):
        """Dokonuje predykcji na nowych danych."""
        return self.model.predict(X)