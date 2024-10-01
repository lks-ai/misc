import os
import arxiv
import openai
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ------------------------- Configuration -------------------------

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the environment variable is set

# Define the arXiv search query for ML, RL, and AI
SEARCH_QUERY = 'cat:cs.AI OR cat:cs.LG OR cat:stat.ML'  # Adjust categories as needed

# Number of papers to fetch
MAX_RESULTS = 500  # Adjust based on your requirements and API limits

# t-SNE parameters
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000

# K-Means parameters
NUM_CLUSTERS = 10  # Adjust based on desired granularity

# ------------------------- Fetching arXiv Papers -------------------------

def fetch_arxiv_papers(query, max_results=100):
    """
    Fetch papers from arXiv based on the search query.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'abstract': result.summary,
            'url': result.entry_id
        })
    return pd.DataFrame(papers)

# ------------------------- Generating Embeddings -------------------------

def get_embeddings(texts, batch_size=20):
    """
    Generate embeddings for a list of texts using OpenAI's API.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            # Optionally, implement retry logic or handle specific exceptions
    return np.array(embeddings)

# ------------------------- Dimensionality Reduction -------------------------

def reduce_dimensions(embeddings, n_components=3, perplexity=30, n_iter=1000):
    """
    Reduce the dimensionality of embeddings using t-SNE.
    """
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(scaled_embeddings)
    return reduced

# ------------------------- Clustering -------------------------

def perform_kmeans(reduced_embeddings, num_clusters=10):
    """
    Perform K-Means clustering on reduced embeddings.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(reduced_embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

# ------------------------- Find Representative Vectors -------------------------

def find_representatives(reduced_embeddings, labels, centroids):
    """
    Find the representative (closest to centroid) for each cluster.
    """
    representatives = []
    for i, centroid in enumerate(centroids):
        cluster_indices = np.where(labels == i)[0]
        cluster_points = reduced_embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representatives.append(closest_index)
    return representatives

# ------------------------- Visualization -------------------------

def plot_interactive_3d(df, reduced_embeddings, labels, centroids, representatives):
    """
    Plot an interactive 3D scatter plot with clusters and centroids.
    """
    fig = go.Figure()

    # Define colors for clusters
    colors = px.colors.qualitative.Dark24
    num_colors = len(colors)

    # Plot each cluster
    for cluster in range(len(centroids)):
        cluster_indices = df[df['cluster'] == cluster].index
        fig.add_trace(go.Scatter3d(
            x=reduced_embeddings[cluster_indices, 0],
            y=reduced_embeddings[cluster_indices, 1],
            z=reduced_embeddings[cluster_indices, 2],
            mode='markers+text',
            name=f'Cluster {cluster+1}',
            marker=dict(
                size=5,
                color=colors[cluster % num_colors],
                opacity=0.8
            ),
            text=df.loc[cluster_indices, 'title'],
            textposition="top center",
            hoverinfo='text'
        ))

    # Plot centroids
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        name='Centroids',
        marker=dict(
            size=10,
            color='black',
            symbol='diamond'
        )
    ))

    # Highlight representatives
    rep_x = reduced_embeddings[representatives, 0]
    rep_y = reduced_embeddings[representatives, 1]
    rep_z = reduced_embeddings[representatives, 2]
    rep_titles = df.loc[representatives, 'title']
    rep_urls = df.loc[representatives, 'url']
    fig.add_trace(go.Scatter3d(
        x=rep_x,
        y=rep_y,
        z=rep_z,
        mode='markers+text',
        name='Representatives',
        marker=dict(
            size=7,
            color='red',
            symbol='star'
        ),
        text=rep_titles,
        textposition="bottom center",
        hoverinfo='text'
    ))

    fig.update_layout(
        title='arXiv ML/RL/AI Papers Clusters',
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        legend=dict(
            itemsizing='constant'
        )
    )

    fig.show()

# ------------------------- Main Workflow -------------------------

def main():
    print("Fetching arXiv papers...")
    df = fetch_arxiv_papers(SEARCH_QUERY, MAX_RESULTS)
    print(f"Fetched {len(df)} papers.")

    # Combine title and abstract for better embeddings
    df['combined'] = df['title'] + ". " + df['abstract']

    print("Generating embeddings...")
    embeddings = get_embeddings(df['combined'].tolist())
    print(f"Generated embeddings of shape: {embeddings.shape}")

    print("Reducing dimensions with t-SNE...")
    reduced = reduce_dimensions(embeddings, n_components=3, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_ITERATIONS)
    print("Dimensionality reduction completed.")

    print("Performing K-Means clustering...")
    labels, centroids = perform_kmeans(reduced, NUM_CLUSTERS)
    df['cluster'] = labels
    print("Clustering completed.")

    print("Finding representative papers for each cluster...")
    representatives = find_representatives(reduced, labels, centroids)
    print("Representatives identified.")

    print("Plotting interactive 3D chart...")
    plot_interactive_3d(df, reduced, labels, centroids, representatives)
    print("Visualization completed.")

if __name__ == "__main__":
    main()

