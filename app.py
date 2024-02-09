# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans

# # Set the page configuration for a more visually appealing layout
# st.set_page_config(page_title="BBC News Article Clusters", layout="wide", page_icon="🗞")

# def scrape_bbc_news():
#     """Scrapes BBC News for article titles and links."""
#     url = 'https://www.bbc.co.uk/news'
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     article_tags = soup.find_all('a', {'class': 'gs-c-promo-heading'})

#     articles = []
#     for tag in article_tags[:20]:  # Limit to 20 articles for simplicity
#         title = tag.text.strip()
#         link = tag['href']
#         if not link.startswith('http'):
#             link = f'https://www.bbc.co.uk{link}'
#         articles.append({'title': title, 'link': link})
   
#     return articles

# def cluster_articles(articles, n_clusters=5):
#     """Clusters the articles based on the similarity of their titles."""
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform([article['title'] for article in articles])

#     if X.shape[0] < n_clusters:
#         n_clusters = X.shape[0]  # Adjust the number of clusters if necessary
   
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(X)
   
#     order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
#     terms = vectorizer.get_feature_names_out()
   
#     clusters = {}
#     for i in range(n_clusters):
#         cluster_terms = [terms[ind] for ind in order_centroids[i, :10]]  # Top-10 terms per cluster
#         clusters[i] = {'articles': [], 'terms': ', '.join(cluster_terms)}
       
#     for i, label in enumerate(kmeans.labels_):
#         clusters[label]['articles'].append(articles[i])
       
#     return clusters

# def show_clusters(n_clusters):
#     """Displays the clustered articles with enhanced UI design."""
#     articles = scrape_bbc_news()
#     if articles:
#         clusters = cluster_articles(articles, n_clusters=n_clusters)
#         for cluster_id, cluster_info in clusters.items():
#             with st.container():
#                 st.markdown(f"### Cluster {cluster_id + 1}: *{cluster_info['terms']}*")
#                 for article in cluster_info['articles']:
#                     st.markdown(f"- [{article['title']}]({article['link']})", unsafe_allow_html=True)
#     else:
#         st.error("No articles found. Please check the URL or try a different site.")

# # Streamlit UI Components
# st.title('🗞 Clustered News Articles from BBC News')
# st.markdown("### Explore how news articles are clustered based on their content.")

# # Sidebar for user input
# with st.sidebar:
#     st.header("Customize Your Analysis")
#     n_clusters = st.slider('Select number of clusters', min_value=2, max_value=20, value=5, step=1)
#     if st.button('Scrape and Cluster Articles'):
#         show_clusters(n_clusters)






import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set the page configuration
st.set_page_config(page_title="BBC News Article Clusters", layout="wide", page_icon="🗞")

def scrape_bbc_news():
    """Scrapes BBC News for article titles and links."""
    url = 'https://www.bbc.co.uk/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_tags = soup.find_all('a', {'class': 'gs-c-promo-heading'})

    articles = []
    for tag in article_tags[:20]:  # Limit to 20 articles
        title = tag.text.strip()
        link = tag['href']
        if not link.startswith('http'):
            link = f'https://www.bbc.co.uk{link}'
        articles.append({'title': title, 'link': link})
    return articles

def cluster_articles(articles, n_clusters=5):
    """Clusters the articles based on the similarity of their titles."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([article['title'] for article in articles])

    if X.shape[0] < n_clusters:
        n_clusters = X.shape[0]  # Adjust the number of clusters if necessary

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    clusters = {}
    for i in range(n_clusters):
        cluster_terms = [terms[ind] for ind in order_centroids[i, :10]]  # Top-10 terms per cluster
        clusters[i] = {'articles': [], 'terms': ', '.join(cluster_terms)}

    for i, label in enumerate(kmeans.labels_):
        clusters[label]['articles'].append(articles[i])

    return clusters

def show_clusters(n_clusters):
    """Displays the clustered articles."""
    articles = scrape_bbc_news()
    if articles:
        clusters = cluster_articles(articles, n_clusters=n_clusters)
        for cluster_id, cluster_info in clusters.items():
            st.markdown(f"### Cluster {cluster_id + 1}: *{cluster_info['terms']}*")
            for article in cluster_info['articles']:
                st.markdown(f"- [{article['title']}]({article['link']})", unsafe_allow_html=True)
    else:
        st.error("No articles found. Please check the URL or try a different site.")

# Streamlit UI Components for Input
st.title('🗞 Clustered News Articles Just For You')
st.markdown("### Explore how news articles are clustered based on their content.")

n_clusters = st.number_input('Select number of clusters', min_value=2, max_value=20, value=5, step=1)
if st.button('Scrape and Cluster Articles'):
    show_clusters(n_clusters)
