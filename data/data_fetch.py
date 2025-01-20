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
        max_days = self.years * 365
        start_date = datetime.datetime.now() - datetime.timedelta(days=max_days)
        end_date = datetime.datetime.now()

        # Pobieranie danych partiami, ponieważ API Yahoo Finance ma ograniczenia czasowe
        data_parts = []
        step = datetime.timedelta(weeks=1)
        current_date = start_date
        total_steps = int((end_date - start_date) / step)

        progress_bar = tqdm(total=total_steps, desc="Pobieranie danych", unit="tydzień")

        while current_date < end_date:
            next_date = min(current_date + step, end_date)
            try:
                part_data = yf.download(
                    self.symbol,
                    start=current_date.strftime('%Y-%m-%d'),
                    end=next_date.strftime('%Y-%m-%d'),
                    interval=self.interval,
                    progress=False
                )
                if not part_data.empty:
                    data_parts.append(part_data)
            except Exception as e:
                print(f"Błąd podczas pobierania danych: {e}")

            current_date = next_date
            progress_bar.update(1)

        progress_bar.close()

        # Łączenie wszystkich pobranych części w jeden DataFrame
        if not data_parts:
            raise ValueError(f"Nie udało się pobrać żadnych danych dla symbolu '{self.symbol}'.")

        gold_data = pd.concat(data_parts)

        # Reset indeksu
        gold_data.reset_index(inplace=True)

        # Ustalanie odpowiednich nazw kolumn
        if 'Date' in gold_data.columns:
            gold_data.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
        elif 'Datetime' in gold_data.columns:
            gold_data.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
        else:
            raise ValueError("Pobrane dane mają nieprawidłowy format lub za mało kolumn.")

        # Usuwanie kolumny "Adj Close" jeśli istnieje
        if 'Adj Close' in gold_data.columns:
            gold_data.drop(columns=['Adj Close'], inplace=True)

        # Usuwanie błędnych i brakujących danych
        gold_data['Datetime'] = pd.to_datetime(gold_data['Datetime'], errors='coerce')
        gold_data.dropna(subset=['Datetime'], inplace=True)

        # Zapis danych do pliku
        gold_data.to_csv(self.output_file, index=False)
        print(f"Dane zostały zapisane do pliku: {self.output_file}")

        return gold_data


# Przykładowe użycie
data_fetcher = GoldDataFetcher(output_file="gold_hourly_data_transformed.csv", years=2)
gold_data = data_fetcher.fetch_data()
print(gold_data.head())