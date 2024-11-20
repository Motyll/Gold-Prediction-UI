from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

class Model:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=15000, learning_rate=0.01, max_depth=8, random_state=30)

    def train(self, X_train, y_train):
        print("Rozpoczynanie treningu modelu...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print("Błąd średniokwadratowy:", mse)
        return mse

    def predict(self, X):
        return self.model.predict(X)
