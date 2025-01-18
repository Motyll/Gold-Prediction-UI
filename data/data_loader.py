import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """Wczytuje dane z pliku CSV."""
        self.data = pd.read_csv(self.filepath, parse_dates=['Datetime'], index_col='Datetime')
        self.data = self.data[['Close', 'Volume']]

    def prepare_features(self):
        """
        Przygotowuje cechy (features) dla modelu:
        - Poprzednia cena
        - Średnie kroczące
        - Zmienność
        - Zmiana procentowa
        - Dzień tygodnia i godzina
        """
        self.data['Previous Price'] = self.data['Close'].shift(1)
        self.data['2h Average'] = self.data['Close'].rolling(window=2).mean().shift(1)
        self.data['4h Average'] = self.data['Close'].rolling(window=4).mean().shift(1)
        self.data['12h Average'] = self.data['Close'].rolling(window=12).mean().shift(1)
        self.data['24h Average'] = self.data['Close'].rolling(window=24).mean().shift(1)
        self.data['4h Volatility'] = self.data['Close'].rolling(window=4).std().shift(1)
        self.data['Price Change (%)'] = self.data['Close'].pct_change() * 100
        self.data['Day of Week'] = self.data.index.dayofweek
        self.data['Hour'] = self.data.index.hour
        self.data['4h Volume Average'] = self.data['Volume'].rolling(window=4).mean().shift(1)

        self.data.dropna(inplace=True)

        # Tworzymy cechy (X) i etykiety (y)
        X = self.data[['Previous Price', '2h Average', '4h Average', '12h Average',
                       '24h Average', '4h Volatility', 'Price Change (%)',
                       'Day of Week', 'Hour', '4h Volume Average']]
        y = self.data['Close']
        return X, y