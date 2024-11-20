import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath, parse_dates=['Datetime'], index_col='Datetime')
        self.data = self.data[['Close']]

    def prepare_features(self):
        self.data['Previous Price'] = self.data['Close'].shift(1)
        self.data['2h Average'] = self.data['Close'].rolling(window=2).mean().shift(1)
        self.data['4h Average'] = self.data['Close'].rolling(window=4).mean().shift(1)

        self.data.dropna(inplace=True)

        X = self.data[['Previous Price', '2h Average', '4h Average']]
        y = self.data['Close']
        return X, y
