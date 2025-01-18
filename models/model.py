from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

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
        print("Błąd średniokwadratowy (MSE):", mse)
        return mse

    def predict(self, X):
        """Dokonuje predykcji na nowych danych."""
        return self.model.predict(X)