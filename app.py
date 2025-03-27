import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
import streamlit as st
from streamlit.components.v1 import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from numpy.linalg import norm
import requests
from bs4 import BeautifulSoup, Comment
import re
import matplotlib.pyplot as plt
import io
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from openai import OpenAI
import tiktoken
from markdownify import MarkdownConverter as md
import uuid
import atexit
from concurrent.futures import ThreadPoolExecutor
import time

st.set_page_config(
    page_title="SiteFocus Tool - Analiza spójności tematycznej",
    page_icon="🎯",
    initial_sidebar_state="expanded",
)

# Dla równoległego przetwarzania
executor = ThreadPoolExecutor(max_workers=1)  # Limit równoległych requestów

async def process_result(result):
    """Przetwarza pojedynczy wynik crawlowania."""
    if result.success:
        print(f"Sukces dla {result.url}")
        return True
    else:
        print(f"Niepowodzenie dla {result.url}: {result.error_message}")
        return False

# Na początku pliku, gdzie inicjalizujemy inne session_state
if 'crawl_cache' not in st.session_state:
    st.session_state.crawl_cache = {}

if 'url_status_cache' not in st.session_state:
    st.session_state.url_status_cache = {}

# Dodajemy zmienne do śledzenia postępu
if 'crawl_progress' not in st.session_state:
    st.session_state.crawl_progress = {
        'total': 0,
        'completed': 0,
        'status': '',
        'current_domain': ''
    }

if 'embedding_progress' not in st.session_state:
    st.session_state.embedding_progress = {
        'total': 0,
        'completed': 0,
        'status': '',
        'current_url': ''
    }

if 'model_list' not in st.session_state:
    st.session_state.model_list = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model_cache' not in st.session_state:  # Dodajemy cache dla wybranego modelu
    st.session_state.model_cache = None
if 'model_initialized' not in st.session_state:  # Nowa flaga
    st.session_state.model_initialized = False
if 'analysis_results' not in st.session_state:  # Dodajemy cache dla wyników
    st.session_state.analysis_results = {}

# Inicjalizacja kontenerów dla pasków postępu
if 'progress_containers' not in st.session_state:
    st.session_state.progress_containers = {
        'crawl_progress': None,
        'crawl_status': None,
        'embedding_progress': None,
        'embedding_status': None,
        'chunk_progress': None,
        'chunk_status': None
    }

# Dodajmy przycisk do czyszczenia cache crawla obok przycisku czyszczenia cache embeddingów
if st.sidebar.button("Wyczyść cache crawla"):
    st.session_state.crawl_cache = {}
    st.success("Cache crawla został wyczyszczony!")

# Na początku pliku, gdzie inicjalizujemy session_state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': None,
        'jina': None,
        'cohere': None
    }

def clean_text(text):
    """Czyści i formatuje tekst."""
    # Usuwamy nadmiarowe białe znaki
    text = ' '.join(text.split())
    # Usuwamy znaki specjalne (oprócz kropek i przecinków)
    text = re.sub(r'[^\w\s.,]', ' ', text)
    # Zamieniamy wielokrotne spacje na pojedyncze
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def count_tokens(text):
    """Liczy tokeny w tekście używając tiktoken."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")  # Używamy standardowego encodera
        tokens = enc.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Błąd podczas liczenia tokenów: {e}")
        return 0


def crawl_url(url):
    try:
        print(f"\n[DEBUG] Próba pobrania: {url}")
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        
        # Nie akceptujemy kompresji Brotli
        session.headers["Accept-Encoding"] = "gzip, deflate"
        
        response = session.get(url, timeout=30, allow_redirects=False)
        print(f"[DEBUG] Status: {response.status_code}")
        print(f"[DEBUG] Headers: {dict(response.headers)}")
        print(f"[DEBUG] Content type: {response.headers.get('content-type')}")
        
        if response.status_code != 200:
            return response.status_code, None
        
        # Używamy requests.content zamiast .text aby mieć surowe dane
        content = response.content.decode('utf-8', errors='ignore')
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Usuwamy komentarze HTML
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Znajdujemy body
        body = soup.find('body')
        if not body:
            return None, None
            
        # Najpierw usuwamy tylko najbardziej oczywiste elementy
        elements_to_remove = [
            'script', 'style', 'noscript', 'iframe',  # Elementy techniczne
            'head', 'header', 'nav', 'footer',  # Elementy strukturalne
            'img',    # Obrazki
            'svg',    # SVG grafiki
            'picture', # Responsywne obrazki
            'figure',  # Figury z obrazkami
        ]
        
        for tag in elements_to_remove:
            for element in body.find_all(tag):
                element.decompose()
        
        # Usuwamy linki nawigacyjne i puste
        for link in body.find_all('a'):
            text = link.text.strip().lower()
            if not text or text == 'scroll to top' or text == 'do góry' or text == 'skip to content':
                link.decompose()
            else:
                # Zachowujemy tekst linku, ale usuwamy sam tag
                link.replace_with(text)
        
        # Usuwamy wszystkie elementy z selektorami
        selectors = [
            # Nawigacja
            '[role*="navigation"]',
            '[role*="nav"]',
            '[role*="menu"]',
            '[class*="main-menu"]',
            
            # Stopka i ciasteczka
            '[class*="footer"]',
            '[class*="cookie-banner"]',
            '[class*="cookie"]',
            
            # Popupy i modalne
            '[class*="popup"]',
            '[class*="modal"]',
            
            # Skip linki - wszystkie możliwe warianty
            '[class*="skip-link"]',
            '[class*="skip"]',
            '[class*="screen-reader"]',
            '[class*="sr-only"]',
            '[class*="visually-hidden"]',
            '[href="#content"]',
            '[href="#main"]',
            '[href="#main-content"]',
            
            # Sidebar i breadcrumby
            '[class*="sidebar"]',
            '[class*="breadcrumb"]',
            
            # Elementy wizualne
            '[class*="gallery"]',
            '[class*="slider"]',
            '[class*="carousel"]',
            
            # Inne
            '[data-nosnippet]',
        ]
        
        # Usuwamy wszystkie elementy z selektorami
        for selector in selectors:
            for element in body.select(selector):
                element.decompose()
        
        # Wyciągamy cały HTML z body
        html_content = str(body)
        #print(html_content)
        
        # Konwertujemy HTML na Markdown z dodatkowymi opcjami
        converter = md(
            heading_style="ATX",
            #strip_document="STRIP",  # Włączamy strip_document dla usunięcia zbędnych elementów
            #wrap=True,  # Zawijanie tekstu
            #wrap_width=80,  # Szerokość zawijania
            #newline_style="SPACES",  # Używaj spacji zamiast \n
            #escape_asterisks=True,  # Escapuj gwiazdki
            #escape_underscores=True  # Escapuj podkreślenia
        )
        markdown_content = converter.convert(html_content)
        
        # Bardziej agresywne czyszczenie tekstu
        # Usuwamy nadmiarowe białe znaki
        markdown_content = ' '.join(markdown_content.split())
        
        # Usuwamy powtarzające się znaki interpunkcyjne
        markdown_content = re.sub(r'([.,!?])\1+', r'\1', markdown_content)
        
        # Usuwamy zbędne znaki markdown, które mogą zwiększać liczbę tokenów
        markdown_content = re.sub(r'[\*\_\~\`]{2,}', ' ', markdown_content)
        
        # Usuwamy linie zawierające tylko znaki specjalne lub krótkie frazy
        markdown_content = re.sub(r'\b\w{1,2}\b', ' ', markdown_content)
        
        # Usuwamy nadmiarowe spacje po czyszczeniu
        markdown_content = re.sub(r'\s+', ' ', markdown_content).strip()
        
        # Usuwamy typowe elementy stopki, które mogą pozostać
        footer_patterns = [
            r'copyright [\d-]+',
            r'all rights reserved',
            r'terms (of|and) conditions',
            r'privacy policy',
            r'cookie policy'
        ]
        for pattern in footer_patterns:
            markdown_content = re.sub(pattern, '', markdown_content, flags=re.IGNORECASE)
        
        #print(markdown_content)
        
        return response.status_code, markdown_content
        
    except Exception as e:
        print(f"[ERROR] Processing failed for {url}: {str(e)}")
        return None, None

def crawl_urls(urls):
    """Crawluje listę URLi."""
    crawled_pages = {}
    total_tokens = 0
    
    # Inicjalizujemy postęp crawlowania
    st.session_state.crawl_progress['total'] = len(urls)
    st.session_state.crawl_progress['completed'] = 0
    st.session_state.crawl_progress['status'] = 'Rozpoczynam crawlowanie stron...'
    
    # Używamy kontenerów z session_state lub tworzymy nowe jeśli nie istnieją
    if st.session_state.progress_containers['crawl_progress'] is None:
        st.session_state.progress_containers['crawl_progress'] = st.empty()
    if st.session_state.progress_containers['crawl_status'] is None:
        st.session_state.progress_containers['crawl_status'] = st.empty()
    
    progress_container = st.session_state.progress_containers['crawl_progress']
    status_container = st.session_state.progress_containers['crawl_status']
    
    # Wyświetlamy początkowy pasek postępu
    progress_bar = progress_container.progress(0)
    status_container.info(st.session_state.crawl_progress['status'])
    
    # Najpierw sprawdzamy cache
    for i, url in enumerate(urls):
        # Aktualizujemy status
        st.session_state.crawl_progress['status'] = f"Przetwarzanie: {url}"
        update_progress(
            progress_container,
            status_container,
            st.session_state.crawl_progress['completed'] / st.session_state.crawl_progress['total'],
            st.session_state.crawl_progress['status']
        )
        
        if url in st.session_state.crawl_cache:
            text = st.session_state.crawl_cache[url]
            num_tokens = count_tokens(text)
            total_tokens += num_tokens
            print(f"[CACHE] Using cached content for {url} ({num_tokens} tokens)")
            crawled_pages[url] = text
            
            # Aktualizujemy postęp
            st.session_state.crawl_progress['completed'] += 1
            update_progress(
                progress_container,
                status_container,
                st.session_state.crawl_progress['completed'] / st.session_state.crawl_progress['total'],
                st.session_state.crawl_progress['status']
            )
            continue
            
        print(f"Crawling: {url}")
        status_code, text = crawl_url(url)
        
        if status_code == 200 and text:
            num_tokens = count_tokens(text)
            total_tokens += num_tokens
            st.session_state.crawl_cache[url] = text
            crawled_pages[url] = text
            print(f"[OK] {url} - {len(text)} chars, {num_tokens} tokens")
        else:
            print(f"[SKIP] {url} => Status code: {status_code}")
        
        # Aktualizujemy postęp
        st.session_state.crawl_progress['completed'] += 1
        update_progress(
            progress_container,
            status_container,
            st.session_state.crawl_progress['completed'] / st.session_state.crawl_progress['total'],
            st.session_state.crawl_progress['status']
        )
    
    # Aktualizujemy status końcowy
    st.session_state.crawl_progress['status'] = f"Zakończono crawlowanie: {len(crawled_pages)} stron z {len(urls)}"
    update_progress(
        progress_container,
        status_container,
        1.0,  # 100% complete
        st.session_state.crawl_progress['status'],
        success=True
    )
    
    print(f"\nSuccessfully crawled {len(crawled_pages)} pages with status 200 (from {len(urls)} total URLs)")
    print(f"Total tokens: {total_tokens}")
    return crawled_pages

#GO!
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}

# Dodajmy unikalny identyfikator sesji
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if st.sidebar.button("Wyczyść cache embeddingów"):
    st.session_state.embeddings_cache = {}
    # Reset progress containers to avoid state inconsistencies
    st.session_state.progress_containers = {
        'crawl_progress': None,
        'crawl_status': None,
        'embedding_progress': None,
        'embedding_status': None,
        'chunk_progress': None,
        'chunk_status': None
    }
    # Reset embedding progress
    st.session_state.embedding_progress = {
        'total': 0,
        'completed': 0,
        'status': '',
        'current_url': ''
    }
    st.success("Cache został wyczyszczony!")

def get_embeddings(text, provider="ollama"):
    """Get embeddings using selected provider."""
    start_time = time.time()
    cache_key = f"{st.session_state.session_id}_{text}"
    
    # Check cache
    if cache_key in st.session_state.embeddings_cache:
        return st.session_state.embeddings_cache[cache_key]
    embedding = None
    max_retries = 3
    retry_delay = 2  # sekundy
    
    try:
        provider = provider.lower()
        
        if provider == "ollama":
            for attempt in range(max_retries):
                try:
                    request_start = time.time()
                    response = requests.post(
                        f"{st.session_state.host.rstrip('/')}/api/embed",
                        json={
                            'model': st.session_state.selected_model,
                            'input': text
                        }
                    )
                    request_time = time.time() - request_start
                    response.raise_for_status()
                    data = response.json()                 
                    # Check for 'embedding' (singular) which is the correct key in Ollama API
                    if 'embedding' in data:
                        embedding = np.array(data['embedding'])
                        break
                    # Also check for 'embeddings' (plural) as a fallback
                    elif 'embeddings' in data and len(data['embeddings']) > 0:
                        embedding = np.array(data['embeddings'][0])
                        break
                    else:
                        print(f"[ERROR] No embedding found in Ollama response. Keys: {list(data.keys())}")
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {error_type} - {error_msg}")
                    # If it's a requests exception, try to get more details
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"[ERROR] Response status: {e.response.status_code}")
                        print(f"[ERROR] Response content: {e.response.text}")
                    if attempt < max_retries - 1:
                        print(f"[INFO] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    raise  # Re-raise the last exception if all retries failed
            
        elif provider == "openai":
            request_start = time.time()
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {st.session_state.api_keys['openai']}"
                },
                json={
                    "input": text,
                    "model": st.session_state.selected_model
                }
            )
            request_time = time.time() - request_start
            response.raise_for_status()
            data = response.json()
            
            
            if 'data' in data and len(data['data']) > 0:
                embedding = np.array(data['data'][0]['embedding'])
        
        elif provider == "jina":
            request_start = time.time()
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {st.session_state.api_keys['jina']}"
                },
                json={
                    "model": st.session_state.selected_model,
                    "dimensions": 1024,
                    "normalized": True,
                    "embedding_type": "float",
                    "input": [{"text": text}] if "clip" in st.session_state.selected_model else [text]
                }
            )
            request_time = time.time() - request_start
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                embedding = np.array(data['data'][0]['embedding'])
        
        elif provider == "cohere":
            for attempt in range(max_retries):
                try:
                    request_start = time.time()
                    response = requests.post(
                        "https://api.cohere.com/v2/embed",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {st.session_state.api_keys['cohere']}"
                        },
                        json={
                            "model": st.session_state.selected_model,
                            "texts": [text],
                            "input_type": "search_document",
                            "embedding_types": ["float"]  # Adding the required parameter
                        }
                    )
                    request_time = time.time() - request_start
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Debug: Print the full response structure
                    #print(f"[DEBUG] Cohere API response structure: {list(data.keys())}")
                    
                    # Check if 'embeddings' exists
                    if 'embeddings' in data:
                        #print(f"[DEBUG] Embeddings type: {type(data['embeddings'])}")
                        
                        # Handle case where embeddings is a dictionary with 'float' key (Cohere format)
                        if isinstance(data['embeddings'], dict) and 'float' in data['embeddings'] and len(data['embeddings']['float']) > 0:
                            embedding = np.array(data['embeddings']['float'][0])
                        # Handle case where embeddings is a list (standard format)
                        elif isinstance(data['embeddings'], list) and len(data['embeddings']) > 0:
                            embedding = np.array(data['embeddings'][0])
                        else:
                            print(f"[ERROR] Embeddings field has unexpected format: {type(data['embeddings'])}")
                    else:
                        print(f"[ERROR] No 'embeddings' field in response. Available keys: {list(data.keys())}")
                        if 'message' in data:
                            print(f"[ERROR] API message: {data['message']}")
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {error_type} - {error_msg}")
                    # If it's a requests exception, try to get more details
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"[ERROR] Response status: {e.response.status_code}")
                        print(f"[ERROR] Response content: {e.response.text}")
                    if attempt < max_retries - 1:
                        print(f"[INFO] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    raise  # Re-raise the last exception if all retries failed
        
        if embedding is not None:
            # Cache the result
            st.session_state.embeddings_cache[cache_key] = embedding
            # Calculate total time
            total_time = time.time() - start_time
            return embedding
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[ERROR] {provider} API error: {error_type} - {error_msg}")
        # If it's a requests exception, try to get more details
        if hasattr(e, 'response') and e.response is not None:
            print(f"[ERROR] Response status: {e.response.status_code}")
            print(f"[ERROR] Response content: {e.response.text}")
        
        total_time = time.time() - start_time
        
        return None

def fetch_sitemap_urls_from_xml(sitemap_url, domain, recursive=False, processed_sitemaps=None, all_urls=None):
    """Fetch URLs from a sitemap XML file."""
    if processed_sitemaps is None:
        processed_sitemaps = set()
    if all_urls is None:
        all_urls = set()
    
    # Jeśli ta sitemap była już przetworzona, pomijamy
    if sitemap_url in processed_sitemaps:
        print(f"[SKIP] Już przetworzono: {sitemap_url}")
        return all_urls
    
    processed_sitemaps.add(sitemap_url)
    print(f"\n--- Przetwarzanie nowej sitemap: {sitemap_url} ---")
    
    try:
        response = requests.get(
            sitemap_url, 
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }, 
            timeout=10
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "lxml-xml")
        
        if soup.find_all("sitemap"):
            print("Znaleziono zagnieżdżone sitemapy:")
            for sitemap in soup.find_all("sitemap"):
                loc = sitemap.find("loc")
                if loc:
                    nested_url = loc.text
                    if nested_url not in processed_sitemaps:
                        print(f"- {nested_url}")
                        if recursive:
                            fetch_sitemap_urls_from_xml(nested_url, domain, recursive=True, 
                                                      processed_sitemaps=processed_sitemaps, 
                                                      all_urls=all_urls)
        else:
            new_urls = 0
            for loc in soup.find_all("loc"):
                url = loc.text
                if not re.search(r"\.(jpg|jpeg|png|gif|svg|webp|bmp|tif|tiff)$", url, re.IGNORECASE):
                    if url not in all_urls:
                        all_urls.add(url)
                        new_urls += 1
            print(f"Dodano {new_urls} nowych URLi (łącznie: {len(all_urls)})")
            
    except Exception as e:
        print(f"Błąd podczas przetwarzania {sitemap_url}: {str(e)}")
    
    return all_urls

def fetch_sitemap_urls(domain):
    """Fetch and parse URLs from sitemaps, excluding images and handling nested sitemaps."""
    domain = domain.replace("https://", "").replace("http://", "").strip("/")
    sitemap_urls = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://{domain}/console/integration/execute/name/GoogleSitemap",
        f"https://{domain}/robots.txt"
    ]
    processed_sitemaps = set()
    all_urls = set()

    print(f"\nRozpoczynam przetwarzanie map witryny dla: {domain}")
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, headers={"User-Agent": "SiteFocusTool/1.0"}, timeout=10)
            response.raise_for_status()
            if "robots.txt" in sitemap_url:
                for line in response.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        nested_sitemap_url = line.split(":", 1)[1].strip()
                        all_urls.update(fetch_sitemap_urls_from_xml(nested_sitemap_url, domain, 
                                                                  recursive=True, 
                                                                  processed_sitemaps=processed_sitemaps,
                                                                  all_urls=all_urls))
            else:
                all_urls.update(fetch_sitemap_urls_from_xml(sitemap_url, domain, 
                                                          recursive=True,
                                                          processed_sitemaps=processed_sitemaps,
                                                          all_urls=all_urls))
        except requests.RequestException:
            continue
            
    print(f"\nZnaleziono łącznie {len(all_urls)} unikalnych URLi")
    return list(all_urls)

def clean_text_from_url(url, domain):
    """Clean URL by removing root domain and extracting readable text."""
    domain = domain.replace("https://", "").replace("http://", "").strip("/")
    url = url.replace(f"https://{domain}/", "").replace(f"http://{domain}/", "")
    text = re.sub(r"[^\w\s]", " ", url)
    text = text.replace("/", " ").replace("_", " ").replace("-", " ")
    return text.strip()

def calculate_site_focus_and_radius(embeddings, reference_embedding=None):
    """Oblicza Site Focus Score i Site Radius względem reference_embedding lub centroidu."""
    if reference_embedding is not None:
        # Używamy reference_embedding jako punktu odniesienia
        site_embedding = reference_embedding
    else:
        # Obliczamy centroid jako site_embedding
        site_embedding = np.mean(embeddings, axis=0)
        site_embedding = site_embedding / np.linalg.norm(site_embedding)
    
    # Obliczamy odległości od punktu odniesienia
    deviations = []
    similarities = []
    for embedding in embeddings:
        # Normalizujemy embedding
        embedding_normalized = embedding / np.linalg.norm(embedding)
        # Obliczamy podobieństwo cosinusowe
        similarity = cosine_similarity(
            site_embedding.reshape(1, -1), 
            embedding_normalized.reshape(1, -1)
        )[0][0]
        similarities.append(similarity)
        # Zamieniamy na odległość
        distance = 1 - similarity
        deviations.append(distance)
    
    deviations = np.array(deviations)
    similarities = np.array(similarities)
    
    # Site Radius - średnia odległość od punktu odniesienia
    site_radius = np.mean(deviations)
    
    # Site Focus Score - miara skupienia na temacie reference URL
    mean_similarity = np.mean(similarities)
    similarity_variance = np.var(similarities)
    site_focus_score = mean_similarity * (1 - similarity_variance)
    
    return site_focus_score, site_radius, site_embedding, deviations

def plot_gradient_strip_with_indicator(score, title):
    """Visualize the score as a gradient strip with an indicator."""
    plt.figure(figsize=(8, 1))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    plt.imshow(gradient, aspect="auto", cmap="RdYlGn_r")
    plt.axvline(x=score * 256, color="black", linestyle="--", linewidth=2)
    plt.gca().set_axis_off()
    plt.title(f"{title}: {score * 100:.2f}%")
    st.pyplot(plt)

def plot_3d_tsne(embeddings, urls, centroid, deviations, normalize=True):
    """Interactive 3D t-SNE scatter plot with hover labels."""
    # Wywołujemy wszystkie trzy rozwiązania, jedno po drugim
    plot_3d_tsne_solution1(embeddings, urls, centroid, deviations, normalize)
    plot_3d_tsne_solution2(embeddings, urls, centroid, deviations, normalize)
    #plot_3d_tsne_solution3(embeddings, urls, centroid, deviations, normalize)

def plot_3d_tsne_solution1(embeddings, urls, centroid, deviations, normalize=True):
    """
    Rozwiązanie 1: Centroid jako średnia punktów po transformacji t-SNE.
    Najpierw wykonujemy t-SNE na danych, a następnie obliczamy centroid w przestrzeni 3D.
    """
    # Konwertujemy do numpy array jeśli nie jest
    embeddings_array = np.array(embeddings)
    
    print(f"[DEBUG] Embeddings shape for 3D (Solution 1): {embeddings_array.shape}")
    
    # Obsługa różnych kształtów embeddings
    if len(embeddings_array.shape) == 3:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    # Opcjonalna normalizacja
    if normalize:
        # Normalizacja L2 embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / norms
    
    # Wykonujemy t-SNE tylko na danych (bez centroidu)
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_results = tsne.fit_transform(embeddings_array)
    
    # Obliczamy centroid w przestrzeni 3D jako średnią punktów po transformacji
    centroid_tsne = np.mean(tsne_results, axis=0)

    fig = px.scatter_3d(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        z=tsne_results[:, 2],
        color=deviations,
        color_continuous_scale="RdYlGn_r",
        hover_name=urls,
        labels={"color": "Deviation"},
        title="Rozwiązanie 1: Centroid jako średnia punktów po transformacji t-SNE"
    )
    fig.add_scatter3d(
        x=[centroid_tsne[0]],
        y=[centroid_tsne[1]],
        z=[centroid_tsne[2]],
        mode="markers",
        marker=dict(
            color='blue',
            symbol='diamond',
            size=20
        ),
        name="Centroid (średnia po t-SNE)"
    )
    st.plotly_chart(fig)

def plot_3d_tsne_solution2(embeddings, urls, centroid, deviations, normalize=True):
    """
    Rozwiązanie 2: Użycie PCA zamiast t-SNE.
    PCA lepiej zachowuje globalne relacje odległości.
    """
    from sklearn.decomposition import PCA
    
    # Konwertujemy do numpy array jeśli nie jest
    embeddings_array = np.array(embeddings)
    centroid_array = np.array(centroid)
    
    print(f"[DEBUG] Embeddings shape for 3D (Solution 2): {embeddings_array.shape}")
    print(f"[DEBUG] Centroid shape for 3D (Solution 2): {centroid_array.shape}")
    
    # Obsługa różnych kształtów embeddings
    if len(embeddings_array.shape) == 3:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    # Obsługa różnych kształtów centroidu
    if len(centroid_array.shape) == 1:
        centroid_array = centroid_array.reshape(1, -1)
    elif len(centroid_array.shape) == 3:
        centroid_array = centroid_array.reshape(centroid_array.shape[0], -1)
    
    # Opcjonalna normalizacja
    if normalize:
        # Normalizacja L2 embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / norms
        
        # Normalizacja L2 centroidu
        centroid_norm = np.linalg.norm(centroid_array, axis=1, keepdims=True)
        centroid_array = centroid_array / centroid_norm
    
    # Łączymy dane z centroidem
    all_data = np.vstack([embeddings_array, centroid_array])
    
    # Wykonujemy PCA
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(all_data)
    
    # Oddzielamy wyniki dla danych i centroidu
    data_pca = pca_results[:-1]
    centroid_pca = pca_results[-1]

    fig = px.scatter_3d(
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        z=data_pca[:, 2],
        color=deviations,
        color_continuous_scale="RdYlGn_r",
        hover_name=urls,
        labels={"color": "Deviation"},
        title="Rozwiązanie 2: Użycie PCA zamiast t-SNE"
    )
    fig.add_scatter3d(
        x=[centroid_pca[0]],
        y=[centroid_pca[1]],
        z=[centroid_pca[2]],
        mode="markers",
        marker=dict(
            color='blue',
            symbol='diamond',
            size=20
        ),
        name="Centroid (PCA)"
    )
    st.plotly_chart(fig)

def plot_3d_tsne_solution3(embeddings, urls, centroid, deviations, normalize=True):
    """
    Rozwiązanie 3: Wizualizacja t-SNE z oryginalnym podejściem.
    Używamy t-SNE do wizualizacji, dodając centroid do danych przed transformacją.
    """
    # Konwertujemy do numpy array jeśli nie jest
    embeddings_array = np.array(embeddings)
    centroid_array = np.array(centroid)
    
    print(f"[DEBUG] Embeddings shape for 3D (Solution 3): {embeddings_array.shape}")
    print(f"[DEBUG] Centroid shape for 3D (Solution 3): {centroid_array.shape}")
    
    # Obsługa różnych kształtów embeddings
    if len(embeddings_array.shape) == 3:
        # Jeśli mamy kształt (n, 1, d) lub (n, d, 1)
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    # Obsługa różnych kształtów centroidu
    if len(centroid_array.shape) == 1:
        centroid_array = centroid_array.reshape(1, -1)
    elif len(centroid_array.shape) == 3:
        centroid_array = centroid_array.reshape(centroid_array.shape[0], -1)
    
    # Opcjonalna normalizacja
    if normalize:
        # Normalizacja L2 embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / norms
        
        # Normalizacja L2 centroidu
        centroid_norm = np.linalg.norm(centroid_array, axis=1, keepdims=True)
        centroid_array = centroid_array / centroid_norm
    
    # Wykonujemy t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_results = tsne.fit_transform(np.vstack([embeddings_array, centroid_array]))
    centroid_tsne = tsne_results[-1]
    tsne_results = tsne_results[:-1]

    fig = px.scatter_3d(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        z=tsne_results[:, 2],
        color=deviations,
        color_continuous_scale="RdYlGn_r",
        hover_name=urls,
        labels={"color": "Deviation"},
        title="Rozwiązanie 3: Oryginalne podejście (centroid dodany przed t-SNE)"
    )
    fig.add_scatter3d(
        x=[centroid_tsne[0]],
        y=[centroid_tsne[1]],
        z=[centroid_tsne[2]],
        mode="markers",
        marker=dict(
            color='blue',
            symbol='diamond',
            size=20
        ),
        name="Centroid (oryginalny)"
    )
    st.plotly_chart(fig)

def plot_spherical_distances_optimized(deviations, embeddings, urls):
    """Improved scatter plot showing distances in a spherical layout."""
    num_points = len(deviations)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    fig = px.scatter_polar(
        r=deviations,
        theta=np.degrees(angles),
        color=deviations,
        color_continuous_scale="RdYlGn_r",
        title="Optimized Spherical Plot of Page Distances from Centroid",
        labels={"color": "Deviation"}
    )
    fig.update_traces(
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
        text=urls
    )
    st.plotly_chart(fig)

def analyze_thematic_center(valid_urls, deviations, embeddings):
    """Analyze thematic center of the website."""
    st.header("🎯 Analiza centrum tematycznego")
    
    # 1. Podsumowanie statystyczne
    st.subheader("📊 Statystyki Centrum")
    col1, col2, col3 = st.columns(3)
    
    mean_distance = np.mean([1 - dev for dev in deviations])
    with col1:
        st.metric("Średnia odległość od centrum", f"{mean_distance:.3f}")
    with col2:
        st.metric("Liczba stron", f"{len(valid_urls)}")
    with col3:
        st.metric("Odchylenie standardowe", f"{np.std([1 - dev for dev in deviations]):.3f}")
    
    # 2. Wykres dystrybucji
    st.subheader("📈 Rozkład odległości od centrum")
    plot_distance_distribution(deviations)
    
    # 3. Lista WSZYSTKICH stron z odległościami
    st.subheader("Lista stron")
    st.markdown("""
    Lista wszystkich stron wraz z ich odległością od centrum tematycznego.
    Im wyższy procent, tym bardziej strona jest reprezentatywna dla głównego tematu.
    """)
    
    # Tworzymy DataFrame ze WSZYSTKIMI stronami
    all_pages_df = pd.DataFrame({
        "URL": valid_urls,
        "Bliskość do centrum": [1 - dev for dev in deviations]
    })
    
    # Formatujemy procenty i sortujemy malejąco
    all_pages_df["Bliskość do centrum"] = all_pages_df["Bliskość do centrum"].apply(lambda x: f"{x*100:.1f}%")
    all_pages_df = all_pages_df.sort_values("Bliskość do centrum", ascending=False)
    
    # Dodajemy numerację
    all_pages_df.index = range(1, len(all_pages_df) + 1)
    all_pages_df.index.name = "Rank"
    
    # Wyświetlamy cały DataFrame
    st.dataframe(all_pages_df)
    
    # 4. Wskazówki interpretacyjne
    st.markdown("""
    ### 💡 Jak interpretować wyniki:
    
    1. **Strony centralne** - Strony z najwyższym procentem bliskości do centrum najlepiej reprezentują główny temat witryny
    2. **Rozkład odległości** - Wykres pokazuje, jak rozproszone są strony wokół centrum tematycznego:
        - Wąski rozkład → Strona jest bardzo spójna tematycznie
        - Szeroki rozkład → Strona pokrywa różnorodne tematy
    3. **Średnia odległość** - Im bliższa 1, tym bardziej spójna tematycznie jest strona
    """)

# Dodajemy nową funkcję do obliczania odległości od URL referencyjnego
def calculate_distances_from_reference(reference_embedding, embeddings, reference_url, valid_urls):
    """Calculate distances from reference URL embedding to all other embeddings."""
    distances = []
    urls = []
    
    # Dodajemy sam URL referencyjny z odległością 0
    distances.append(0)
    urls.append(reference_url)
    
    # Obliczamy odległości dla pozostałych URLi
    for emb, url in zip(embeddings, valid_urls):
        if url != reference_url:  # Pomijamy URL referencyjny w pętli
            similarity = np.dot(reference_embedding, emb)
            distance = 1 - similarity
            distances.append(distance)
            urls.append(url)
            
    return np.array(distances), urls

def collect_closest_pages(domain, urls, distances, n=10):
    """Collect n closest pages from each domain."""
    df = pd.DataFrame({
        "Domain": [domain] * len(urls),
        "URL": urls,
        "Distance": distances
    })
    return df.nlargest(n, "Distance")

def plot_spherical_reference_comparison(reference_url, domains_data):
    """Create spherical plot with reference URL in center and closest pages around."""
    fig = go.Figure()
    
    # Dodajemy punkt centralny (URL referencyjny)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=[reference_url],
        name='Reference URL'
    ))
    
    # Generujemy punkty na sferze dla każdej domeny
    colors = px.colors.qualitative.Set3  # Różne kolory dla różnych domen
    for idx, (domain, df) in enumerate(domains_data.groupby("Domain")):
        # Generujemy punkty na sferze
        n_points = len(df)
        phi = np.linspace(0, 2*np.pi, n_points)
        theta = np.linspace(-np.pi/2, np.pi/2, n_points)
        
        # Konwertujemy odległości na promień (im mniejsza odległość, tym bliżej centrum)
        r = 1 + df["Distance"].values
        
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(size=6, color=colors[idx % len(colors)]),
            text=df["URL"],
            name=domain,
            hovertemplate="Domain: %{text}<br>Distance: %{customdata}<extra></extra>",
            customdata=df["Distance"]
        ))
    
    fig.update_layout(
        title="Spherical Distribution of Closest Pages to Reference URL",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        showlegend=True
    )
    
    st.plotly_chart(fig)

def collect_cross_domain_analysis(reference_embedding, domain_results, n=10):
    """Analyze results across all domains with reference URL as center."""
    all_results = []
    
    for domain_data in domain_results:
        domain = domain_data['domain']
        embeddings = domain_data['embeddings']
        urls = domain_data['urls']
        
        # Obliczamy odległości od URL referencyjnego
        distances = []
        for emb in embeddings:
            similarity = np.dot(reference_embedding, emb)
            distance = 1 - similarity
            distances.append(distance)
        
        # Zbieramy n najbliższych stron
        closest_indices = np.argsort(distances)[:n]
        for idx in closest_indices:
            all_results.append({
                'Domain': domain,
                'URL': urls[idx],
                'Distance': distances[idx]
            })
    
    return pd.DataFrame(all_results)

def plot_distance_distribution(deviations):
    """Plot distribution of distances from center."""
    fig = go.Figure()
    
    # Konwertujemy deviacje na procenty bliskości do centrum
    proximities = [(1 - dev) * 100 for dev in deviations]
    
    # Dodajemy histogram
    fig.add_trace(go.Histogram(
        x=proximities,
        nbinsx=30,
        name='Rozkład',
        hovertemplate="Bliskość do centrum: %{x:.1f}%<br>Liczba stron: %{y}<extra></extra>"
    ))
    
    # Dodajemy linię średniej
    mean_proximity = np.mean(proximities)
    fig.add_vline(
        x=mean_proximity,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Średnia: {mean_proximity:.1f}%",
        annotation_position="top"
    )
    
    # Aktualizujemy layout
    fig.update_layout(
        title="Rozkład odległości stron od centrum tematycznego",
        xaxis_title="Bliskość do centrum (%)",
        yaxis_title="Liczba stron",
        showlegend=False,
        bargap=0.1
    )
    
    # Wyświetlamy wykres
    st.plotly_chart(fig)

def plot_2d_tsne(embeddings, urls, centroid, deviations, normalize=True):
    """2D t-SNE scatter plot with hover labels."""
    # Konwertujemy do numpy array jeśli nie jest
    embeddings_array = np.array(embeddings)
    centroid_array = np.array(centroid)
    
    print(f"[DEBUG] Embeddings shape: {embeddings_array.shape}")
    print(f"[DEBUG] Centroid shape: {centroid_array.shape}")
    
    # Obsługa różnych kształtów embeddings
    if len(embeddings_array.shape) == 3:
        # Jeśli mamy kształt (n, 1, d) lub (n, d, 1)
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    # Obsługa różnych kształtów centroidu
    if len(centroid_array.shape) == 1:
        centroid_array = centroid_array.reshape(1, -1)
    elif len(centroid_array.shape) == 3:
        centroid_array = centroid_array.reshape(centroid_array.shape[0], -1)
    
    # Opcjonalna normalizacja
    if normalize:
        # Normalizacja L2 embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / norms
        
        # Normalizacja L2 centroidu
        centroid_norm = np.linalg.norm(centroid_array, axis=1, keepdims=True)
        centroid_array = centroid_array / centroid_norm
    
    # Dodajemy centroid do embeddings
    all_embeddings = np.vstack([embeddings_array, centroid_array])
    
    # Wykonujemy t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_results = tsne.fit_transform(all_embeddings)
    
    # Oddzielamy wyniki dla stron i centroidu
    pages_tsne = tsne_results[:-1]
    centroid_tsne = tsne_results[-1]
    
    # Tworzymy DataFrame dla plotly
    df = pd.DataFrame({
        'x': pages_tsne[:, 0],
        'y': pages_tsne[:, 1],
        'Distance': deviations,
        'URL': urls
    })
    
    # Tworzymy wykres
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Distance',
        color_continuous_scale='RdYlGn_r',
        hover_data=['URL'],
        title='2D t-SNE Projection of Pages',
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
    )
    
    # Dodajemy centroid
    fig.add_trace(
        go.Scatter(
            x=[centroid_tsne[0]],
            y=[centroid_tsne[1]],
            mode='markers',
            marker=dict(
                color='red',
                symbol='star',
                size=20
            ),
            name='Centroid',
            hoverinfo='name'
        )
    )
    
    # Aktualizujemy layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    
    # Wyświetlamy wykres
    st.plotly_chart(fig)

def get_focus_score_interpretation(score):
    """Zwraca interpretację dla Site Focus Score."""
    if score < 0.30:
        return "🔴 Niska spójność tematyczna - strona porusza wiele różnych tematów"
    elif score < 0.60:
        return "🟡 Średnia spójność tematyczna - strona ma kilka głównych obszarów tematycznych"
    else:
        return "🟢 Wysoka spójność tematyczna - strona jest mocno skoncentrowana na jednym temacie"

def get_radius_interpretation(radius):
    """Zwraca interpretację dla Site Radius."""
    if radius < 0.15:
        return "🟢 Małe rozproszenie - treści są bardzo spójne ze sobą"
    elif radius < 0.30:
        return "🟡 Średnie rozproszenie - treści są umiarkowanie zróżnicowane"
    else:
        return "🔴 Duże rozproszenie - treści są bardzo zróżnicowane"

def split_into_chunks(text, provider="ollama"):
    """Dzieli tekst na chunki z różnymi limitami dla różnych dostawców."""
    # Tokenizacja tekstu
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    total_tokens = len(tokens)
    
    # Ustawiamy limit tokenów i overlap w zależności od dostawcy
    if provider == "openai" or provider == "jina":
        max_tokens = 8000
        overlap = 1200  # 15% z 8000
    else:  # ollama, cohere i inne
        max_tokens = 500
        overlap = 75
    
    # Jeśli tekst jest krótszy niż max_tokens, zwracamy go jako jeden chunk
    if total_tokens <= max_tokens:
        chunk_text = enc.decode(tokens)
        print(f"[CHUNK 1/1] Rozmiar: {total_tokens} tokenów")
        print(f"[TOKENS] Tekst zawiera {total_tokens} tokenów, podzielono na 1 chunków")
        print(f"[ŚREDNIA] Średnio {total_tokens:.1f} tokenów na chunk")
        print(f"[CHUNKS] Podzielono tekst na 1 chunków")
        return [chunk_text]
    
    # Dla dłuższych tekstów, dzielimy na chunki z overlapem
    chunks = []
    # Obliczamy krok (step) - ile tokenów przesuwamy się w każdej iteracji
    step = max_tokens - overlap
    
    # Dzielimy na chunki
    for i in range(0, total_tokens, step):
        # Określamy początek i koniec chunka
        start = i
        end = min(i + max_tokens, total_tokens)
        
        # Wycinamy chunk
        chunk = tokens[start:end]
        if chunk:
            chunk_text = enc.decode(chunk)
            chunks.append(chunk_text)
            # Logowanie
            print(f"[CHUNK {len(chunks)}/{(total_tokens + step - 1) // step}] Rozmiar: {len(chunk)} tokenów")
    
    # Logowanie podsumowania
    print(f"[TOKENS] Tekst zawiera {total_tokens} tokenów, podzielono na {len(chunks)} chunków")
    if chunks:
        print(f"[ŚREDNIA] Średnio {total_tokens / len(chunks):.1f} tokenów na chunk")
    print(f"[CHUNKS] Podzielono tekst na {len(chunks)} chunków")
    
    return chunks

def update_progress(container, status_container, progress, status, success=False, error=False):
    """Aktualizuje pasek postępu i status."""
    if container is not None:
        container.progress(progress)
    
    if status_container is not None:
        if success:
            status_container.success(status)
        elif error:
            status_container.error(status)
        else:
            status_container.info(status)

def optimize_text_for_embedding(text):
    """Optymalizuje tekst przed generowaniem embeddingów, aby zmniejszyć liczbę tokenów."""
    start_time = time.time()
    
    # Usuwamy nadmiarowe białe znaki
    whitespace_start = time.time()
    text = ' '.join(text.split())
    whitespace_time = time.time() - whitespace_start
    # Usuwamy powtarzające się znaki interpunkcyjne
    punctuation_start = time.time()
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    punctuation_time = time.time() - punctuation_start  
    # Usuwamy typowe elementy stopki, które mogą pozostać
    footer_start = time.time()
    footer_patterns = [
        r'copyright [\d-]+',
        r'all rights reserved',
        r'terms (of|and) conditions',
        r'privacy policy',
        r'cookie policy',
        r'kontakt',
        r'contact us',
        r'newsletter',
        r'subscribe',
        r'follow us',
        r'social media',
        r'share this',
        r'comments',
        r'related posts',
        r'read more'
    ]
    
    for pattern in footer_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    footer_time = time.time() - footer_start  
    # Usuwamy nadmiarowe spacje po czyszczeniu
    final_spaces_start = time.time()
    text = re.sub(r'\s+', ' ', text).strip()
    final_spaces_time = time.time() - final_spaces_start

    # Calculate total time and token reduction
    total_time = time.time() - start_time
    final_tokens = count_tokens(text)
    return text

def get_averaged_embedding(text, provider="ollama", url=None):
    """Generuje uśredniony embedding z chunków tekstu."""
    start_time = time.time()
    
    # Aktualizujemy status embeddingu w UI
    if url:
        st.session_state.embedding_progress['current_url'] = url
        st.session_state.embedding_progress['status'] = f"Generowanie embeddingu dla: {url}"
    
    # Używamy kontenerów z session_state lub tworzymy nowe jeśli nie istnieją
    if st.session_state.progress_containers['embedding_progress'] is None:
        st.session_state.progress_containers['embedding_progress'] = st.empty()
    if st.session_state.progress_containers['embedding_status'] is None:
        st.session_state.progress_containers['embedding_status'] = st.empty()
    
    progress_container = st.session_state.progress_containers['embedding_progress']
    status_container = st.session_state.progress_containers['embedding_status']
    
    # Wyświetlamy początkowy status
    if url:
        progress_container.progress(0)
        status_container.info(f"Generowanie embeddingu dla: {url}")

    # Najpierw optymalizujemy tekst, aby zmniejszyć liczbę tokenów
    if url:
        status_container.info(f"Optymalizacja tekstu dla: {url}")
    
    optimization_start = time.time()
    text = optimize_text_for_embedding(text)
    optimization_time = time.time() - optimization_start
    
    # Dzielimy na chunki
    if url:
        status_container.info(f"Dzielenie tekstu na chunki dla: {url}")
    
    chunking_start = time.time()
    chunks = split_into_chunks(text, provider=provider)
    chunking_time = time.time() - chunking_start
    print(f"[CHUNKS] Podzielono tekst na {len(chunks)} chunków")
    
    # Jeśli mamy tylko jeden chunk, nie musimy uśredniać
    if len(chunks) == 1:
        print("[EMBEDDING] Generuję embedding dla pojedynczego chunka...")
        single_embedding = get_embeddings(chunks[0], provider=provider)
        return single_embedding
    chunk_embeddings = []
    successful_chunks = 0
    failed_chunks = 0
    
    for i, chunk in enumerate(chunks, 1):
        print(f"[CHUNK {i}/{len(chunks)}] Generuję embedding...")
        embedding = get_embeddings(chunk, provider=provider)
        if embedding is not None:
            chunk_embeddings.append(embedding)
            successful_chunks += 1
        else:
            failed_chunks += 1
    # Uśredniamy embeddingi i normalizujemy    
    averaging_start = time.time()
    averaged_embedding = np.mean(chunk_embeddings, axis=0)
    averaged_embedding = averaged_embedding / np.linalg.norm(averaged_embedding)
    averaging_time = time.time() - averaging_start
    print(f"[EMBEDDING] Wygenerowano uśredniony embedding z {len(chunk_embeddings)} chunków")
    return averaged_embedding

# Streamlit Interface
st.title("SiteFocus Tool")

# Sprawdzenie dostępności Ollamy
try:
    response = requests.get('http://localhost:11434/api/tags')
    ollama_available = response.status_code == 200
    if ollama_available:
        st.success("✅ Wykryto lokalną instalację Ollamy")
except requests.exceptions.ConnectionError:
    ollama_available = False
    st.warning("⚠️ Nie wykryto lokalnej instalacji Ollamy")

# Wybór dostawcy embeddingów
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    provider = st.radio(
        "Wybierz dostawcę:",
        options=["Ollama", "OpenAI", "Jina", "Cohere"],
        index=0,
    ).lower()

with col2:
    if provider == "ollama":
        st.session_state.host = st.text_input(
            "Host:",
            value="http://localhost:11434/",
            help="Domyślnie: http://localhost:11434/"
        )
    elif provider == "openai":
        st.session_state.host = "https://api.openai.com/v1/"
        st.write(f"Endpoint: {st.session_state.host}")
    elif provider == "jina":
        st.session_state.host = "https://api.jina.ai/v1/"
        st.write(f"Endpoint: {st.session_state.host}")
    elif provider == "cohere":
        st.session_state.host = "https://api.cohere.com/v2/"
        st.write(f"Endpoint: {st.session_state.host}")

with col3:
    if provider in ["openai", "jina", "cohere"]:
        api_key = st.text_input(
            "Klucz API:",
            type="password",
            value=st.session_state.api_keys.get(provider),
            help=f"Wprowadź klucz API dla {provider.upper()}"
        )
        if api_key:
            st.session_state.api_keys[provider] = api_key
    else:
        st.write("API: -")

# Guzik do pobierania modeli
if st.button("Pobierz modele"):
    print(f"\n--- Pobieranie modeli dla {provider} ---")
    
    # Reset poprzedniego wyboru modelu
    st.session_state.selected_model = None
    st.session_state.model_initialized = False
    
    if provider == "ollama":
        try:
            host_url = st.session_state.host.rstrip('/')
            print(f"Próba połączenia z Ollama na URL: {host_url}")
            response = requests.get(f'{host_url}/api/tags')
            print(f"Status odpowiedzi: {response.status_code}")
            
            if response.status_code == 200:
                models = response.json()
                if models.get('models'):
                    st.session_state.model_list = [model['name'] for model in models['models']]
                    print(f"Znaleziono modele: {st.session_state.model_list}")
                    st.success("Pobrano listę modeli")
                else:
                    st.session_state.model_list = []
                    print("Brak modeli w odpowiedzi")
                    st.info("Brak dostępnych modeli")
            else:
                st.error(f"Błąd podczas pobierania modeli: {response.status_code}")
        except Exception as e:
            st.error(f"Błąd podczas pobierania modeli Ollama: {str(e)}")
            
    elif provider == "openai":
        if not st.session_state.api_keys.get('openai'):
            st.error("Wprowadź klucz API OpenAI")
        else:
            st.session_state.model_list = ["text-embedding-3-small", "text-embedding-3-large"]
            st.success("Pobrano listę modeli")
            
    elif provider == "jina":
        if not st.session_state.api_keys.get('jina'):
            st.error("Wprowadź klucz API Jina")
        else:
            st.session_state.model_list = ["jina-clip-v2", "jina-embeddings-v3"]
            st.success("Pobrano listę modeli")
            
    elif provider == "cohere":
        if not st.session_state.api_keys.get('cohere'):
            st.error("Wprowadź klucz API Cohere")
        else:
            try:
                # Próba pobrania listy modeli z API
                response = requests.get(
                    "https://api.cohere.com/v1/models",
                    headers={
                        "Authorization": f"Bearer {st.session_state.api_keys['cohere']}"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Filtrowanie tylko modeli embeddingowych
                embedding_models = [model['name'] for model in data.get('models', []) 
                                   if 'embed' in model.get('name', '').lower()]
                
                if embedding_models:
                    st.session_state.model_list = embedding_models
                    st.success(f"Pobrano {len(embedding_models)} modeli Cohere")
                else:
                    # Fallback do hardcodowanej listy
                    st.session_state.model_list = [
                        "embed-english-v3.0", 
                        "embed-english-light-v3.0", 
                        "embed-multilingual-v3.0", 
                        "embed-multilingual-light-v3.0"
                    ]
                    st.warning("Nie znaleziono modeli embeddingowych. Używam domyślnej listy.")
            except Exception as e:
                print(f"[ERROR] Nie udało się pobrać listy modeli Cohere: {str(e)}")
                # Fallback do hardcodowanej listy
                st.session_state.model_list = [
                    "embed-english-v3.0", 
                    "embed-english-light-v3.0", 
                    "embed-multilingual-v3.0", 
                    "embed-multilingual-light-v3.0"
                ]
                st.warning("Nie udało się pobrać listy modeli. Używam domyślnej listy.")

# W miejscu gdzie wybieramy model
if st.session_state.model_list:
    if not st.session_state.model_initialized or st.session_state.selected_model not in st.session_state.model_list:
        # Inicjalizujemy model tylko raz lub gdy poprzedni model nie jest dostępny
        st.session_state.selected_model = st.session_state.model_list[0]
        st.session_state.model_initialized = True
    
    # Wyświetlamy selectbox z aktualnie wybranym modelem
    selected = st.selectbox(
        "Wybierz model:", 
        st.session_state.model_list,
        index=st.session_state.model_list.index(st.session_state.selected_model)
    )
    
    # Aktualizujemy model tylko jeśli użytkownik faktycznie zmienił wybór
    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        print(f"Zmieniono model na: {selected}")


# Debug mode toggle and debug panel


# Najpierw pole URL referencyjnego
reference_url = st.text_input(
    "URL referencyjny (opcjonalnie):", 
    placeholder="https://example.com/page",
    help="Jeśli nie podasz URL referencyjnego, analiza będzie przeprowadzona względem centrum tematycznego wszystkich stron."
)

# Następnie pole na domeny
domains = st.text_area(
    "Wprowadź domeny (każda w nowej linii):", 
    placeholder="example.com\nexample2.com\nexample3.com",
    help="Wprowadz jedną lub więcej domen, każdą w nowej linii"
)

# Inicjalizacja reference_embedding
reference_embedding = None

st.markdown("""
    ℹ️ **Bliskość do centrum** - miara pokazująca jak blisko centrum tematycznego znajduje się dana strona:
    - Wyższa wartość = strona jest bardziej zgodna z główną tematyką domeny
    - Niższa wartość = strona bardziej odbiega od typowej treści na stronie
""")


if st.button("START"):
    # Sprawdzamy konfigurację na początku
    if provider == "ollama" and not st.session_state.selected_model:
        if st.session_state.model_cache:
            st.session_state.selected_model = st.session_state.model_cache
        else:
            st.error("Nie wybrano modelu! Proszę kliknąć 'Pobierz modele' i wybrać model.")
            st.stop()
    elif provider == "openai" and not st.session_state.api_keys.get('openai'):
        st.error("Nie podano klucza API OpenAI!")
        st.stop()
    elif provider == "jina" and not st.session_state.api_keys.get('jina'):
        st.error("Nie podano klucza API Jina!")
        st.stop()
    elif provider == "cohere" and not st.session_state.api_keys.get('cohere'):
        st.error("Nie podano klucza API Cohere!")
        st.stop()

    if domains:
        # Najpierw crawlujemy URL referencyjny (jeśli podany)
        if reference_url:
            ref_crawled = crawl_urls([reference_url])
            if ref_crawled and reference_url in ref_crawled:
                reference_text = ref_crawled[reference_url]
                reference_embedding = get_averaged_embedding(reference_text, provider=provider)
                if reference_embedding is not None:
                    reference_embedding = reference_embedding / norm(reference_embedding)
                    st.success("Pomyślnie wygenerowano embedding dla URL referencyjnego")
                else:
                    st.error("Nie udało się wygenerować embeddingu dla URL referencyjnego")
            
        # Dzielimy tekst na listę domen i usuwamy puste linie
        domain_list = [d.strip() for d in domains.split('\n') if d.strip()]
        
        # Na samym początku
        all_results = []  # Lista na wszystkie wyniki
        
        # Główna pętla po domenach
        for domain in domain_list:
            st.subheader(f"Analiza domeny: {domain}")
            print(f"Rozpoczęcie przetwarzania domeny: {domain}")
            
            with st.spinner(f"Fetching URLs for {domain}..."):
                urls = fetch_sitemap_urls(domain)
                if not urls:
                    st.error(f"No URLs found for {domain}")
                    continue
                    
                st.info(f"Found {len(urls)} URLs for {domain}")
                
                # Sprawdzamy statusy i crawlujemy od razu strony z 200
                crawled_pages = crawl_urls(urls)
                if not crawled_pages:
                    st.warning(f"No valid pages found from {len(urls)} URLs")
                    continue
                    
                valid_urls = list(crawled_pages.keys())
                texts = list(crawled_pages.values())
                st.info(f"Successfully crawled {len(crawled_pages)} pages with status 200 (from {len(urls)} total URLs)")
                
                # Generowanie embeddingów
                embeddings = []
                valid_urls = []  # Resetujemy listę
                
                # Inicjalizujemy postęp embeddingów
                st.session_state.embedding_progress['total'] = len(crawled_pages)
                st.session_state.embedding_progress['completed'] = 0
                st.session_state.embedding_progress['status'] = 'Rozpoczynam generowanie embeddingów...'
                
                # Nagłówek sekcji embeddingów
                st.subheader("Generowanie embeddingów")
                
                # Używamy kontenerów z session_state lub tworzymy nowe jeśli nie istnieją
                if st.session_state.progress_containers['embedding_progress'] is None:
                    st.session_state.progress_containers['embedding_progress'] = st.empty()
                if st.session_state.progress_containers['embedding_status'] is None:
                    st.session_state.progress_containers['embedding_status'] = st.empty()
                
                progress_container = st.session_state.progress_containers['embedding_progress']
                status_container = st.session_state.progress_containers['embedding_status']
                
                # Wyświetlamy początkowy pasek postępu i status
                progress_container.progress(0)
                status_container.info('Rozpoczynam generowanie embeddingów...')
                
                for url, text in crawled_pages.items():
                    print(f"Generuję embedding dla URL: {url}")
                    # Aktualizujemy status
                    st.session_state.embedding_progress['current_url'] = url
                    st.session_state.embedding_progress['status'] = f"Generowanie embeddingu dla: {url}"
                    update_progress(
                        progress_container,
                        status_container,
                        st.session_state.embedding_progress['completed'] / st.session_state.embedding_progress['total'],
                        st.session_state.embedding_progress['status']
                    )
                    
                    # Przekazujemy URL do funkcji get_averaged_embedding
                    embedding = get_averaged_embedding(text, provider=provider, url=url)
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_urls.append(url)
                        
                    # Aktualizujemy postęp
                    st.session_state.embedding_progress['completed'] += 1
                    st.session_state.embedding_progress['status'] = f"Zakończono {st.session_state.embedding_progress['completed']} z {st.session_state.embedding_progress['total']} embeddingów"
                    update_progress(
                        progress_container,
                        status_container,
                        st.session_state.embedding_progress['completed'] / st.session_state.embedding_progress['total'],
                        st.session_state.embedding_progress['status'],
                        success=(st.session_state.embedding_progress['completed'] == st.session_state.embedding_progress['total'])
                    )
                
                # Najpierw obliczamy wyniki
                if embeddings:
                    st.success(f"Successfully generated {len(embeddings)} embeddings with model {st.session_state.selected_model}")
                    
                    # Najpierw obliczamy wyniki
                    if reference_embedding is not None:
                        site_focus_score, site_radius, site_embedding, deviations = calculate_site_focus_and_radius(
                            embeddings, 
                            reference_embedding=reference_embedding
                        )
                    else:
                        site_focus_score, site_radius, site_embedding, deviations = calculate_site_focus_and_radius(
                            embeddings
                        )
                    
                    # Potem zapisujemy do cache
                    st.session_state.analysis_results[domain] = {
                        'embeddings': embeddings,
                        'valid_urls': valid_urls,
                        'site_focus_score': site_focus_score,
                        'site_radius': site_radius,
                        'centroid': site_embedding,
                        'deviations': deviations
                    }
                    
                    # Wyświetlamy wyniki
                    st.subheader("📊 Metryki spójności tematycznej")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Site Focus Score", value=f"{site_focus_score:.2%}")
                        st.markdown(get_focus_score_interpretation(site_focus_score))
                        st.markdown("""
                        **Site Focus Score (Spójność tematyczna)**:

                        - **<30%** - Niska spójność tematyczna  
                        - Strona porusza wiele różnych, niepowiązanych tematów  
                        - Treści są bardzo zróżnicowane  
                        - Typowe dla portali ogólnotematycznych lub agregatorów treści  

                        - **30-60%** - Średnia spójność tematyczna  
                        - Strona ma kilka głównych obszarów tematycznych  
                        - Treści są powiązane, ale zróżnicowane  
                        - Typowe dla portali branżowych lub blogów o szerokiej tematyce  

                        - **>60%** - Wysoka spójność tematyczna  
                        - Strona koncentruje się na jednym głównym temacie  
                        - Treści są ściśle ze sobą powiązane  
                        - Typowe dla specjalistycznych stron i blogów tematycznych  
                        """)
                        
                    with col2:
                        st.metric(label="Site Radius", value=f"{site_radius:.2%}")
                        st.markdown(get_radius_interpretation(site_radius))
                        st.markdown("""
                                    **Site Radius (Rozproszenie treści)**:

                                    - **<15%** - Małe rozproszenie  
                                    - Treści są bardzo spójne ze sobą  
                                    - Poszczególne strony trzymają się głównego tematu  
                                    - Wskazuje na konsekwentną strategię treści  

                                    - **15-30%** - Średnie rozproszenie  
                                    - Treści są umiarkowanie zróżnicowane  
                                    - Występują odstępstwa od głównego tematu  
                                    - Typowe dla stron z różnorodnymi podsekcjami  

                                    - **>30%** - Duże rozproszenie  
                                    - Treści znacząco różnią się od siebie  
                                    - Duże odchylenia od głównego tematu  
                                    - Może wskazywać na brak spójnej strategii treści  
                                    """)
                    # Dodajemy szczegółowe wyjaśnienie skali
                    st.markdown("""💡 **Optymalne wartości** zależą od typu strony i jej przeznaczenia. Dla wyspecjalizowanego bloga tematycznego korzystne będą wysokie wartości Site Focus Score i niskie Site Radius. Dla portalu informacyjnego naturalne będą średnie wartości obu metryk.
                    """)
                    
                    # 2. Szczegółowe wizualizacje z opisami
                    st.markdown("---")
                    st.subheader("siteFocusScore")
                    st.markdown(""" 
                    **Site Focus Score** odzwierciedla, jak mocno treści na stronie są skupione wokół jednego obszaru tematycznego.  
                    Wyższy wynik oznacza większą spójność tematyczną.
                    """)
                    
                    st.markdown(f"""
                        <div style='margin: 20px 0;'>
                            <p style='text-align: center; font-size: 1.2em;'>siteFocusScore: {site_focus_score:.2%}</p>
                            <div style='background: linear-gradient(to right, #00ff00, #ffff00, #ff0000); height: 30px; position: relative; border-radius: 4px;'>
                                <div style='position: absolute; left: {site_focus_score*100}%; border-left: 2px dashed black; height: 100%;'></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("siteRadius")
                    st.markdown("""
                    **Site Radius** mierzy, jak bardzo poszczególne strony odbiegają od głównego tematu strony.  
                    Mniejszy promień oznacza większą spójność treści.
                    """)
                    
                    st.markdown(f"""
                        <div style='margin: 20px 0;'>
                            <p style='text-align: center; font-size: 1.2em;'>siteRadius: {site_radius:.2%}</p>
                            <div style='background: linear-gradient(to right, #00ff00, #ffff00, #ff0000); height: 30px; position: relative; border-radius: 4px;'>
                                <div style='position: absolute; left: {site_radius*100}%; border-left: 2px dashed black; height: 100%;'></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Dodajemy 2D t-SNE visualization
                    st.subheader("2D t-SNE Projection")
                    plot_2d_tsne(embeddings, valid_urls, site_embedding, deviations)
                    
                    # Jeśli mamy URL referencyjny, dodajemy podsumowanie najbliższych stron
                    if reference_embedding is not None:
                        st.subheader("🎯 Top 10 stron najbliższych URL referencyjnemu")
                        
                        # Obliczamy odległości od URL referencyjnego
                        distances_from_ref = []
                        for emb in embeddings:
                            similarity = np.dot(reference_embedding, emb)
                            distance = 1 - similarity
                            distances_from_ref.append(distance)
                        
                        # Tworzymy DataFrame z wynikami
                        domain_results_df = pd.DataFrame({
                            'URL': valid_urls,
                            'Distance': distances_from_ref
                        })
                        
                        # Sortujemy i bierzemy top 10
                        domain_top_10 = domain_results_df.nsmallest(10, 'Distance')
                        
                        # Dodajemy numerację
                        domain_top_10.index = range(1, len(domain_top_10) + 1)
                        domain_top_10.index.name = "Rank"
                        
                        # Formatujemy odległości na procenty podobieństwa
                        domain_top_10['Similarity'] = domain_top_10['Distance'].apply(lambda x: f"{(1-x)*100:.1f}%")
                        
                        # Wyświetlamy tabelę
                        st.dataframe(domain_top_10[['URL', 'Similarity']])
                        
                        # Dodajemy krótkie podsumowanie
                        mean_similarity = (1 - np.mean(distances_from_ref)) * 100
                        st.markdown(f"""
                        **Podsumowanie dla domeny {domain}:**
                        - Średnie podobieństwo do URL referencyjnego: {mean_similarity:.1f}%
                        - Liczba przeanalizowanych stron: {len(valid_urls)}
                        """)
                    
                    # Analiza centrum tematycznego
                    analyze_thematic_center(valid_urls, deviations, embeddings)
                    
                    # Wizualizacje 3D
                    st.header("🌐 Wizualizacje 3D")
                    st.subheader("3D t-SNE Projection")
                    plot_3d_tsne(embeddings, valid_urls, site_embedding, deviations)
                    
                    st.subheader("Spherical Distance Plot")
                    plot_spherical_distances_optimized(deviations, embeddings, valid_urls)
                    
                    # Zbieranie danych do analizy cross-domain
                    if reference_embedding is not None:
                        for url, emb in zip(valid_urls, embeddings):
                            similarity = np.dot(reference_embedding, emb)
                            distance = 1 - similarity
                            all_results.append({
                                'Domain': domain,
                                'URL': url,
                                'Distance': distance,
                                'Embedding': emb
                            })

        # CAŁKOWICIE POZA PĘTLĄ - analiza cross-domain
        if reference_url and len(all_results) > 0:
            st.header("🌍 Porównanie domen", anchor="porownanie-domen")
            st.subheader("Najbliższe strony względem URL referencyjnego")
            
            # Tworzymy DataFrame ze wszystkich wyników
            cross_domain_results = pd.DataFrame(all_results)
            
            # Przekształcamy reference_embeddings na właściwy kształt
            if len(reference_embedding.shape) == 1:
                reference_embedding = reference_embedding.reshape(1, -1)
            reference_centroid = reference_embedding
            
            # Przeliczamy odległości dla wszystkich URLi
            url_distances = {}
            for _, row in cross_domain_results.iterrows():
                url = row['URL']
                embeddings = np.array(row['Embedding'])
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
                distance = 1 - cosine_similarity(reference_centroid, embeddings)[0][0]
                url_distances[url] = distance
            
            # Aktualizujemy DataFrame o nowe odległości
            cross_domain_results['New_Distance'] = cross_domain_results['URL'].map(url_distances)
            
            # Sortujemy po nowych odległościach
            results_table = (cross_domain_results
                .sort_values('New_Distance')
                .groupby('Domain')
                .head(10)
                .reset_index(drop=True))
            
            # Tworzymy wiersz dla URL referencyjnego
            reference_row = pd.DataFrame([{
                'Domain': urlparse(reference_url).netloc,
                'URL': reference_url,
                'New_Distance': 0.0
            }])
            
            # Łączymy URL referencyjny z wynikami
            results_table = pd.concat([reference_row, results_table], ignore_index=True)
            
            # Wyświetlamy tabelę z nowymi odległościami
            st.dataframe(
                results_table[['Domain', 'URL', 'New_Distance']],
                column_config={
                    "Domain": "Domena",
                    "URL": "Adres URL",
                    "New_Distance": st.column_config.NumberColumn(
                        "Odległość od centroidu",
                        format="%.3f",
                    )
                },
                hide_index=True
            )
            
            # Dodajemy wykres polarny używając zoptymalizowanej funkcji
            st.markdown("---")
            st.subheader("🎯 Optimized Spherical Plot of Page Distances from Centroid")
            
            plot_spherical_distances_optimized(
                deviations=results_table['New_Distance'].values,
                embeddings=None,  # Nie potrzebujemy embeddingów do tego wykresu
                urls=results_table['URL'].values
            )

            st.markdown("""
            ### 🎯 Interpretacja wykresu polarnego:
            - Środek wykresu reprezentuje URL referencyjny (odległość = 0)
            - Odległość od środka pokazuje różnicę tematyczną:
                * Bliżej środka = treść bardziej podobna do referencyjnej
                * Dalej od środka = większa różnica w treści
            - Kolor punktów reprezentuje odległość (zielony = blisko, czerwony = daleko)
            """)

        # Po wygenerowaniu embeddingów
        if domain in st.session_state.analysis_results:
            results = st.session_state.analysis_results[domain]
            # Wyświetlamy zapisane wyniki...
            st.success(f"Using cached results for {domain} with {len(results['embeddings'])} embeddings")
            # ... reszta wyświetlania wyników ...

    # Czyszczenie po zakończeniu
    def cleanup():
        # Funkcja czyszcząca zasoby po zakończeniu
        pass
    atexit.register(cleanup)

if st.sidebar.button("Wyczyść cache wyników"):
    st.session_state.analysis_results = {}
    st.success("Cache wyników został wyczyszczony")

# Zamiast tego możemy dodać informację o aktualnej sesji
st.sidebar.info(f"ID sesji: {st.session_state.session_id[:8]}...")

if st.sidebar.button("Wyczyść klucze API"):
    st.session_state.api_keys = {
        'openai': None,
        'jina': None,
        'cohere': None
    }
    st.success("Klucze API zostały wyczyszczone!")
