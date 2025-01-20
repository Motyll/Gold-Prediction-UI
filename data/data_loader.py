import pandas as pd


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        # Wczytanie danych z pliku CSV
        try:
            self.data = pd.read_csv(self.filepath, parse_dates=['Datetime'], index_col='Datetime')
            if self.data.empty:
                raise ValueError(f"Plik {self.filepath} jest pusty lub dane są niekompletne.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Plik {self.filepath} nie został znaleziony.")
        except Exception as e:
            raise ValueError(f"Wystąpił błąd podczas wczytywania pliku {self.filepath}: {e}")

        # Wybór interesujących kolumn
        required_columns = ['Close', 'Volume']
        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(f"Brak wymaganej kolumny: {column}")
        self.data = self.data[required_columns]

    def prepare_features(self):
        # Tworzenie dodatkowych cech na podstawie danych historycznych
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

        # Usuwanie brakujących wartości
        self.data.dropna(inplace=True)

        # Przygotowanie danych wejściowych (X) i wyjściowych (y)
        feature_columns = [
            'Previous Price', '2h Average', '4h Average', '12h Average',
            '24h Average', '4h Volatility', 'Price Change (%)',
            'Day of Week', 'Hour', '4h Volume Average'
        ]
        X = self.data[feature_columns]
        y = self.data['Close']
        return X, y