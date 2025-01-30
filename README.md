# SiteFocus Tool 🎯

Narzędzie do analizy spójności tematycznej stron internetowych wykorzystujące embeddingi tekstu.

## 🌟 Funkcje

- Analiza spójności tematycznej stron internetowych
- Wsparcie dla wielu dostawców embeddingów (Ollama, OpenAI, Jina)
- Automatyczne crawlowanie stron z sitemap
- Inteligentne czyszczenie treści (usuwanie menu, stopek, reklam)
- Wizualizacja wyników
- Cache dla crawlowanych stron i embeddingów

## 📋 Wymagania

- Python 3.8+
- Ollama (opcjonalnie dla lokalnych embeddingów)
- Klucz API OpenAI (opcjonalnie)
- Klucz API Jina (opcjonalnie)

## 🚀 Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/username/sitefocus.git
cd sitefocus
```

2. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

3. (Opcjonalnie) Zainstaluj i uruchom Ollamę:
- [Instrukcje instalacji Ollamy](https://ollama.ai/download)

## 🎮 Użycie

1. Uruchom aplikację:
```bash
streamlit run app.py
```

2. Wybierz dostawcę embeddingów (Ollama/OpenAI/Jina)
3. Wprowadź URL referencyjny (opcjonalnie)
4. Wprowadź listę domen do analizy
5. Kliknij START

## 📊 Metryki

- **Site Focus Score** - Miara spójności tematycznej (0-100%)
  - <30% - Niska spójność
  - 30-60% - Średnia spójność
  - >60% - Wysoka spójność

## 🔧 Konfiguracja

- Ollama: Domyślnie `http://localhost:11434/`
- OpenAI: Wymaga klucza API
- Jina: Wymaga klucza API

## 📝 Licencja

MIT License

## 👥 Autorzy

- [Roman Rozenberger](https://rozenberger.com)