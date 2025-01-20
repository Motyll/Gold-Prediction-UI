from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from catboost import CatBoostRegressor

class Model:
    def __init__(self, model_type="GradientBoosting"):
        if model_type == "GradientBoosting":
            self.model = GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.01, max_depth=8, random_state=42
            )
        elif model_type == "RandomForest":
            self.model = RandomForestRegressor(
                n_estimators=500, max_depth=10, random_state=42
            )
        elif model_type == "ExtraTrees":
            self.model = ExtraTreesRegressor(
                n_estimators=500, max_depth=None, random_state=42
            )
        # elif model_type == "CatBoost":
        #     self.model = CatBoostRegressor(
        #         iterations=1000, learning_rate=0.01, depth=8, random_seed=42, verbose=0
        #     )
        else:
            raise ValueError(f"Nieobsługiwany typ modelu: {model_type}")

    # trenowanie modelu na danych treningowych
    def train(self, X_train, y_train):
        print("Rozpoczynanie treningu modelu...")
        self.model.fit(X_train, y_train)

    # ewaluacja modelu na danych testowych
    def evaluate(self, X_test, y_test):
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

    # predykcja na nowych danych
    def predict(self, X):
        return self.model.predict(X)