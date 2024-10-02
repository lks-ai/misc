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
import plotly.express as px
import logging
import sys

# ------------------------- Configuration -------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the environment variable is set

if not openai.api_key:
    logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Define the arXiv search query for ML, RL, and AI
SEARCH_QUERY = 'cat:cs.AI OR cat:cs.LG OR cat:stat.ML'  # Adjust categories as needed

# Number of papers to fetch
MAX_RESULTS = 100  # Start with a smaller number for testing

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
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for result in search.results():
            papers.append({
                'title': result.title.strip().replace('\n', ' '),
                'abstract': result.summary.strip().replace('\n', ' '),
                'url': result.entry_id
            })
        logging.info(f"Fetched {len(papers)} papers from arXiv.")
        return pd.DataFrame(papers)
    except Exception as e:
        logging.error(f"Error fetching papers from arXiv: {e}")
        return pd.DataFrame()

# ------------------------- Generating Embeddings -------------------------

import openai

def get_embeddings(texts, batch_size=20):
    """
    Generate embeddings for a list of texts using OpenAI's API.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = openai.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            logging.info(f"Generated embeddings for batch {i//batch_size + 1} / {len(texts)//batch_size + 1}")
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API error during embedding generation: {e}")
            # Optionally, implement retry logic here
        except Exception as e:
            logging.error(f"Unexpected error during embedding generation: {e}")
    return np.array(embeddings)

# ------------------------- Dimensionality Reduction -------------------------

def reduce_dimensions(embeddings, n_components=3, perplexity=30, n_iter=1000):
    """
    Reduce the dimensionality of embeddings using t-SNE.
    """
    try:
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        logging.info("Embeddings scaled successfully.")

        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
        reduced = tsne.fit_transform(scaled_embeddings)
        logging.info("Dimensionality reduction with t-SNE completed.")
        return reduced
    except Exception as e:
        logging.error(f"Error during dimensionality reduction: {e}")
        return None

# ------------------------- Clustering -------------------------

def perform_kmeans(reduced_embeddings, num_clusters=10):
    """
    Perform K-Means clustering on reduced embeddings.
    """
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(reduced_embeddings)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        logging.info("K-Means clustering completed.")
        return labels, centroids
    except Exception as e:
        logging.error(f"Error during K-Means clustering: {e}")
        return None, None

# ------------------------- Find Representative Vectors -------------------------

def find_representatives(reduced_embeddings, labels, centroids):
    """
    Find the representative (closest to centroid) for each cluster.
    """
    representatives = []
    for i, centroid in enumerate(centroids):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            logging.warning(f"Cluster {i} has no members.")
            representatives.append(None)
            continue
        cluster_points = reduced_embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representatives.append(closest_index)
    logging.info("Representative papers for each cluster identified.")
    return representatives

# ------------------------- Visualization -------------------------

def plot_interactive_3d(df, reduced_embeddings, labels, centroids, representatives):
    """
    Plot an interactive 3D scatter plot with clusters and centroids.
    """
    try:
        fig = go.Figure()

        # Define colors for clusters
        colors = px.colors.qualitative.Dark24
        num_colors = len(colors)

        # Plot each cluster
        for cluster in range(len(centroids)):
            cluster_indices = df[df['cluster'] == cluster].index
            if cluster_indices.empty:
                continue
            fig.add_trace(go.Scatter3d(
                x=reduced_embeddings[cluster_indices, 0],
                y=reduced_embeddings[cluster_indices, 1],
                z=reduced_embeddings[cluster_indices, 2],
                mode='markers',
                name=f'Cluster {cluster+1}',
                marker=dict(
                    size=5,
                    color=colors[cluster % num_colors],
                    opacity=0.8
                ),
                hovertext=df.loc[cluster_indices, 'title'],
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
        rep_indices = [idx for idx in representatives if idx is not None]
        rep_x = reduced_embeddings[rep_indices, 0]
        rep_y = reduced_embeddings[rep_indices, 1]
        rep_z = reduced_embeddings[rep_indices, 2]
        rep_titles = df.loc[rep_indices, 'title']
        rep_urls = df.loc[rep_indices, 'url']
        fig.add_trace(go.Scatter3d(
            x=rep_x,
            y=rep_y,
            z=rep_z,
            mode='markers+text',
            name='Representatives',
            marker=dict(
                size=7,
                color='red',
                symbol='diamond'  # Changed from 'star' to 'diamond'
            ),
            text=rep_titles,
            textposition="top center",
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
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        fig.show()
        logging.info("Interactive 3D chart displayed successfully.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")

# ------------------------- Main Workflow -------------------------

def main():
    logging.info("Starting arXiv Skimmer...")

    # Step 1: Fetch arXiv papers
    df = fetch_arxiv_papers(SEARCH_QUERY, MAX_RESULTS)
    if df.empty:
        logging.error("No papers fetched. Exiting the script.")
        sys.exit(1)

    # Combine title and abstract for better embeddings
    df['combined'] = df['title'] + ". " + df['abstract']

    # Step 2: Generate embeddings
    logging.info("Generating embeddings...")
    embeddings = get_embeddings(df['combined'].tolist())
    logging.info(f"Generated embeddings of shape: {embeddings.shape}")

    if embeddings.size == 0:
        logging.error("No embeddings generated. Exiting the script.")
        sys.exit(1)

    # Step 3: Dimensionality Reduction
    logging.info("Reducing dimensions with t-SNE...")
    reduced = reduce_dimensions(embeddings, n_components=3, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_ITERATIONS)
    if reduced is None:
        logging.error("Dimensionality reduction failed. Exiting the script.")
        sys.exit(1)

    # Step 4: Clustering
    logging.info("Performing K-Means clustering...")
    labels, centroids = perform_kmeans(reduced, NUM_CLUSTERS)
    if labels is None or centroids is None:
        logging.error("Clustering failed. Exiting the script.")
        sys.exit(1)
    df['cluster'] = labels

    # Step 5: Find Representatives
    logging.info("Finding representative papers for each cluster...")
    representatives = find_representatives(reduced, labels, centroids)

    # Step 6: Visualization
    logging.info("Plotting interactive 3D chart...")
    plot_interactive_3d(df, reduced, labels, centroids, representatives)

    logging.info("arXiv Skimmer completed successfully.")

if __name__ == "__main__":
    main()

