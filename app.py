import streamlit as st
from streamlit.components.v1 import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from numpy.linalg import norm
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import io
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

#GO!

if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}

if st.sidebar.button("Wyczyść cache embeddingów"):
    st.session_state.embeddings_cache = {}
    st.success("Cache został wyczyszczony!")

@st.cache_data
def get_embeddings(text):
    """Get embeddings using Ollama API with caching"""
    # Sprawdzamy cache
    cache_key = text
    if cache_key in st.session_state.embeddings_cache:
        print(f"Używam embeddingu z cache dla tekstu: {text[:100]}...")
        return st.session_state.embeddings_cache[cache_key]
    
    try:
        print(f"Generowanie embeddingu dla tekstu: {text[:100]}...")
        response = requests.post(
            'http://localhost:11434/api/embed',
            json={
                'model': 'snowflake-arctic-embed2',
                'input': text
            }
        )
        response.raise_for_status()
        data = response.json()
        
        if st.session_state.get('debug', False):
            st.write("API Response:", data)
            
        if 'embeddings' in data and len(data['embeddings']) > 0:
            # Zapisujemy do cache
            embedding = np.array(data['embeddings'][0])
            st.session_state.embeddings_cache[cache_key] = embedding
            print("Embedding wygenerowany pomyślnie")
            return embedding
        else:
            print(f"Błąd: Nieoczekiwana struktura odpowiedzi API: {data}")
            return None
    except Exception as e:
        print(f"Błąd podczas generowania embeddingu: {e}")
        return None

def fetch_sitemap_urls(domain):
    """Fetch and parse URLs from sitemaps, excluding images and handling nested sitemaps."""
    domain = domain.replace("https://", "").replace("http://", "").strip("/")
    sitemap_urls = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://{domain}/console/integration/execute/name/GoogleSitemap",
        f"https://{domain}/robots.txt"
    ]
    all_urls = []

    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, headers={"User-Agent": "SiteFocusTool/1.0"}, timeout=10)
            response.raise_for_status()
            if "robots.txt" in sitemap_url:
                for line in response.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        nested_sitemap_url = line.split(":", 1)[1].strip()
                        all_urls.extend(fetch_sitemap_urls_from_xml(nested_sitemap_url, domain, recursive=True))
            else:
                all_urls.extend(fetch_sitemap_urls_from_xml(sitemap_url, domain, recursive=True))
        except requests.RequestException:
            continue
    return list(set(all_urls))

def fetch_sitemap_urls_from_xml(sitemap_url, domain, recursive=False):
    """Fetch URLs from a sitemap XML file."""
    print(f"\n--- Przetwarzanie sitemap: {sitemap_url} ---")
    urls = []
    try:
        response = requests.get(
            sitemap_url, 
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }, 
            timeout=10
        )
        print(f"Status odpowiedzi: {response.status_code}")
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        print(f"Typ zawartości: {content_type}")
        
        soup = BeautifulSoup(response.content, "lxml-xml")
        
        # Debugowanie zawartości XML
        print(f"Zawartość XML (pierwsze 500 znaków):")
        print(response.text[:500])
        
        if soup.find_all("sitemap"):
            print("Znaleziono zagnieżdżone sitemapy")
            for sitemap in soup.find_all("sitemap"):
                loc = sitemap.find("loc")
                if loc:
                    nested_url = loc.text
                    print(f"Zagnieżdżona sitemap: {nested_url}")
                    if recursive:
                        urls.extend(fetch_sitemap_urls_from_xml(nested_url, domain, recursive=True))
        else:
            for loc in soup.find_all("loc"):
                url = loc.text
                if not re.search(r"\.(jpg|jpeg|png|gif|svg|webp|bmp|tif|tiff)$", url, re.IGNORECASE):
                    urls.append(url)
            print(f"Znaleziono {len(urls)} URLi")
            
    except Exception as e:
        print(f"Błąd podczas przetwarzania {sitemap_url}: {str(e)}")
        print(f"Typ błędu: {type(e).__name__}")
    
    return urls

def clean_text_from_url(url, domain):
    """Clean URL by removing root domain and extracting readable text."""
    domain = domain.replace("https://", "").replace("http://", "").strip("/")
    url = url.replace(f"https://{domain}/", "").replace(f"http://{domain}/", "")
    text = re.sub(r"[^\w\s]", " ", url)
    text = text.replace("/", " ").replace("_", " ").replace("-", " ")
    return text.strip()

def calculate_site_focus_and_radius(embeddings):
    """Oblicza Site Focus Score i promień."""
    centroid = np.mean(embeddings, axis=0)
    deviations = np.array([
        1 - cosine_similarity(embedding.reshape(1, -1), centroid.reshape(1, -1))[0][0]
        for embedding in embeddings
    ])
    # Teraz deviations to "odległości od centrum"
    return np.mean(deviations), np.std(deviations), centroid, deviations

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

def plot_3d_tsne(embeddings, urls, centroid, deviations):
    """Interactive 3D t-SNE scatter plot with hover labels."""
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_results = tsne.fit_transform(np.vstack([embeddings, centroid]))
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
        title="3D t-SNE Projection of Page Embeddings"
    )
    fig.add_scatter3d(
        x=[centroid_tsne[0]],
        y=[centroid_tsne[1]],
        z=[centroid_tsne[2]],
        mode="markers",
        marker=dict(size=15, color="green"),
        name="Centroid"
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

def plot_2d_tsne(embeddings, urls, centroid, deviations):
    """2D t-SNE scatter plot with hover labels."""
    # Dodajemy centroid do embeddings
    all_embeddings = np.vstack([embeddings, centroid])
    
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

# Streamlit Interface
st.title("SiteFocus Tool (Ollama API Version)")


# Debug mode toggle
if 'debug' not in st.session_state:
    st.session_state.debug = False
st.sidebar.checkbox('Debug Mode', value=False, key='debug')

# Test connection to Ollama
try:
    test_embedding = get_embeddings("test")
    if test_embedding is not None:
        st.success("Successfully connected to Ollama")
except Exception as e:
    st.error(f"Could not connect to Ollama. Make sure it's running on localhost:11434\nError: {e}")
    st.stop()

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
    if domains:
        # Generujemy embedding dla URL referencyjnego (jeśli podany)
        if reference_url:
            print(f"Generowanie embeddingu dla URL referencyjnego: {reference_url}")
            reference_text = clean_text_from_url(reference_url, reference_url.split('/')[2])
            reference_embedding = get_embeddings(reference_text)
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
                
                # Przetwarzanie URLi
                cleaned_texts = [clean_text_from_url(url, domain) for url in urls]
                embeddings = []
                valid_urls = []
                
                for idx, (url, text) in enumerate(zip(urls, cleaned_texts)):
                    embedding = get_embeddings(text)
                    if embedding is not None:
                        normalized_embedding = embedding / norm(embedding)
                        embeddings.append(normalized_embedding)
                        valid_urls.append(url)
                
                # Analiza pojedynczej domeny
                if len(embeddings) > 0:
                    embeddings = np.array(embeddings)
                    site_focus_score, site_radius, centroid, deviations = calculate_site_focus_and_radius(embeddings)
                    
                    # 1. Metryki z interpretacją
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
                    plot_2d_tsne(embeddings, valid_urls, centroid, deviations)
                    
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
                    plot_3d_tsne(embeddings, valid_urls, centroid, deviations)
                    
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

