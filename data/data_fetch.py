import yfinance as yf
import pandas as pd
import datetime
from tqdm import tqdm


class GoldDataFetcher:
    def __init__(self, symbol="GC=F", interval="1h", years=2, output_file="gold_hourly_data.csv"):
        self.symbol = symbol
        self.interval = interval
        self.years = years
        self.output_file = output_file

    def fetch_data(self):
        max_days = 730  # 2 lata = 730 dni
        start_date = datetime.datetime.now() - datetime.timedelta(days=max_days)
        end_date = datetime.datetime.now()
        data_parts = []
        step = datetime.timedelta(weeks=1)
        current_date = start_date
        total_steps = int((end_date - start_date) / step)

        progress_bar = tqdm(total=total_steps, desc="Pobieranie danych", unit="tydzień")

        while current_date < end_date:
            next_date = min(current_date + step, end_date)
            part_data = yf.download(self.symbol, start=current_date, end=next_date, interval=self.interval)
            data_parts.append(part_data)
            current_date = next_date
            progress_bar.update(1)

        progress_bar.close()
        gold_data = pd.concat(data_parts)

        if gold_data.empty:
            raise ValueError(f"Nie udało się pobrać danych dla symbolu '{self.symbol}'.")

        gold_data.reset_index(inplace=True)
        gold_data.columns = ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        gold_data['Datetime'] = pd.to_datetime(gold_data['Datetime'], errors='coerce')
        gold_data.dropna(subset=['Datetime'], inplace=True)

        gold_data.to_csv(self.output_file, index=False)
        print(f"Dane zostały zapisane do pliku: {self.output_file}")

        return gold_data


data_fetcher = GoldDataFetcher(output_file="gold_hourly_data_transformed.csv", years=2)
gold_data = data_fetcher.fetch_data()
gold_data.head()