# SiteFocus Tool

Narzędzie do analizy tematycznej stron internetowych z wykorzystaniem modelu embeddingów `snowflake-arctic-embed2` i biblioteki Streamlit.

## Wymagania

Przed rozpoczęciem pracy upewnij się, że masz zainstalowane następujące elementy:
- Python 3.11: python\.org
- Streamlit: streamlit\.io
- Ollama: ollama\.com (do obsługi embeddingów)
- Model embeddingów: `snowflake-arctic-embed2`
- Git git-scm\.com

## Instalacja i konfiguracja

### Pobierz kod źródłowy
Możesz pobrać kod na dwa sposoby:

#### Przez Git (zalecane):
```bash
# Sklonuj repozytorium
git clone https://github.com/romek-rozen/siteFocusOllama.git

# Przejdź do katalogu projektu
cd siteFocusOllama
```

#### Przez przeglądarkę:
1. Wejdź na stronę `https://github.com/romek-rozen/siteFocusOllama`
2. Kliknij zielony przycisk "Code"
3. Wybierz "Download ZIP"
4. Rozpakuj pobrany plik

### Utwórz wirtualne środowisko
Aby zapewnić izolację środowiska, utwórz wirtualne środowisko:

```bash
python -m venv myenv
```

### Aktywuj środowisko

Windows:
```bash
myenv\Scripts\activate
```

Mac/Linux:
```bash
source myenv/bin/activate
```

### Zainstaluj wymagane pakiety
Zainstaluj wszystkie zależności z pliku requirements.txt:
```bash
pip install -r requirements.txt
```

### Skonfiguruj Ollama
Pobierz wymagany model embeddingów:
```bash
ollama pull snowflake-arctic-embed2
```

Uruchom serwer Ollama:
```bash
ollama serve
```

## Uruchamianie narzędzia

1. Upewnij się, że serwer Ollama jest uruchomiony:
```bash
ollama serve
```

2. Uruchom aplikację Streamlit:
```bash
streamlit run app.py
```

3. Otwórz przeglądarkę i przejdź pod wyświetlony adres URL (domyślnie localhost:8501).

## Funkcjonalności

- Analiza tematyczna stron internetowych
- Generowanie embeddingów przy użyciu Ollama API
- Wizualizacje 2D i 3D (t-SNE, wykresy sferyczne)
- Analiza spójności tematycznej z metrykami Site Focus Score i Site Radius

## Czyszczenie cache

Cache embeddingów można wyczyścić w aplikacji za pomocą przycisku "Wyczyść cache embeddingów" w pasku bocznym.

## Debugowanie

Możesz włączyć tryb debugowania, zaznaczając opcję "Debug Mode" w pasku bocznym aplikacji.

---

[Roman Rozenberger](https://rozenberger.com)