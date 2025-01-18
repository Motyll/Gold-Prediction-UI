# **Prognoza Cen Złota**

Aplikacja webowa do przewidywania cen złota na podstawie historycznych danych. Wykorzystuje uczenie maszynowe, aby prognozować ceny na kolejny dzień roboczy, zapewniając czytelne wykresy i tabele z wynikami.

---

## **Funkcjonalności**
1. **Dynamiczny wybór modelu uczenia maszynowego**:
   - Dostępne modele: Gradient Boosting, Random Forest, Extra Trees, CatBoost.
2. **Elastyczne zakresy czasowe danych historycznych**:
   - Wybór danych z ostatnich: 2 lat, 1 roku, 6 miesięcy, 3 miesięcy.
3. **Prognoza cen na kolejny dzień roboczy**:
   - Pomijanie weekendów, gdy giełda jest zamknięta.
4. **Integracja kursów walut**:
   - Prognozy mogą być wyświetlane w USD, EUR, PLN, lub GBP.
5. **Czytelny interfejs**:
   - Prognozy prezentowane w formie tabeli i wykresu wyświetlanego obok siebie.
6. **Automatyczna obsługa danych wejściowych**:
   - Dane historyczne są pobierane serwisu Yahoo Finance.

---

## **Technologie**
- **Python**: Główny język programowania.
- **Streamlit**: Framework do budowy aplikacji webowych.
- **Scikit-learn**: Do implementacji modeli uczenia maszynowego i skalowania danych.
- **Yahoo Finance API**: Pobieranie danych o złocie oraz kursach walut.
- **Matplotlib**: Wizualizacja wyników.
- **Pandas**: Analiza i przetwarzanie danych.
- **NumPy**: Operacje matematyczne.

---
   
## **Struktura projektu**
```
├── catboost_info           # Informacje o modelach CatBoost
├── data
│   ├── data_loader.py      # Moduł do wczytywania i przetwarzania danych
│   ├── data_fetch.py       # Pobieranie danych z Yahoo Finance
├── models
│   ├── model.py            # Definicje modeli uczenia maszynowego
│   ├── predictor.py        # Klasa predyktora do prognozowania cen
├── app.py                  # Główna aplikacja Streamlit
├── requirements.txt        # Lista wymaganych bibliotek
├── README.md               # Dokumentacja projektu
└── gold_hourly_data_transformed.csv # Przetworzone dane historyczne
```

---

## **Uruchomienie**

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/Motyll/Gold-Prediction-UI.git
    cd Gold-Prediction-UI
    ```
2. Zainstaluj wymagane biblioteki:

    ```bash
    pip install -r requirements.txt
    ```
3. Uruchom aplikację:

    ```bash
    streamlit run app.py
    ```
   
---

## **Autorzy**
- **Piotr Jałocha**
- **Michał Kulesza**
- **Mateusz Kotula**