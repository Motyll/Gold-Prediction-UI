import yfinance as yf
import pandas as pd
from tqdm import tqdm
import datetime

symbol = 'GC=F'
period = '1y'
interval = '1h'

start_date = datetime.datetime.now() - pd.DateOffset(years=1)
end_date = datetime.datetime.now()
data_parts = []
step = datetime.timedelta(weeks=1)  # Pobieranie w odstępach tygodniowych
current_date = start_date
total_steps = int((end_date - start_date) / step)
progress_bar = tqdm(total=total_steps, desc="Pobieranie danych", unit="tydzień")

while current_date < end_date:
    next_date = min(current_date + step, end_date)
    part_data = yf.download(symbol, start=current_date, end=next_date, interval=interval)
    data_parts.append(part_data)
    current_date = next_date
    progress_bar.update(1)

progress_bar.close()

# Łączenie wszystkich fragmentów danych
gold_data = pd.concat(data_parts)

if gold_data.empty:
    raise ValueError(f"Nie udało się pobrać danych dla symbolu '{symbol}'.")

# Formatowanie i zapis danych
gold_data.reset_index(inplace=True)
column_names = ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
gold_data.columns = column_names
gold_data['Datetime'] = pd.to_datetime(gold_data['Datetime'], errors='coerce')
gold_data.dropna(subset=['Datetime'], inplace=True)

output_file_path = 'data/gold_hourly_data_transformed.csv'
gold_data.to_csv(output_file_path, index=False)

print(f"Plik został zapisany jako '{output_file_path}' z poprawnym formatowaniem.")
