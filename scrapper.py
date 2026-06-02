import scrapy
import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests 
import re
from transformers import pipeline,  AutoImageProcessor, AutoModel
import os
from os import listdir
import glob
from PIL import Image
import torch
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numpy.typing import NDArray


def scrapper():
    url = "https://raw.communitydragon.org/10.1/plugins/rcp-be-lol-game-data/global/default/assets/characters"
    r = requests.get(url)
    htmldata = r.text
    soup = BeautifulSoup(htmldata, 'lxml')
    directories = soup.find_all('td', 'link')

    for dir in directories[1:]:
        hero = dir.a['href'].replace("/", "")
        if hero[:3] == "tft":
            continue


        url2 = f"https://raw.communitydragon.org/10.1/plugins/rcp-be-lol-game-data/global/default/assets/characters/{hero}/skins/base/{hero}loadscreen.jpg"

        resp = requests.get(url2)
        if resp.status_code != 200:

            urlreplace = f"https://raw.communitydragon.org/10.1/plugins/rcp-be-lol-game-data/global/default/assets/characters/{hero}/skins/base/{hero}loadscreen.{hero}.jpg"
            resp = requests.get(urlreplace)


        with open(f"heros/{hero}.jpg", "wb") as f:
            f.write(resp.content)



def fetch_champions_with_faction_and_splash():
    universe_url = (
        "https://universe-meeps.leagueoflegends.com/v1/en_gb/search/index.json"
    )

    ddragon_url = (
        "https://ddragon.leagueoflegends.com/cdn/15.12.1/data/en_US/champion.json"
    )

    universe_data = requests.get(universe_url).json()
    ddragon_data = requests.get(ddragon_url).json()["data"]

    name_to_id = {
        info["name"]: champ_id
        for champ_id, info in ddragon_data.items()
    }

    champions = universe_data.get("champions", [])

    champion_info = []

    for champ in champions:
        name = champ.get("name")

        faction = champ.get("associated-faction-slug")

        if not faction:
            faction = "runeterra"

        champion_id = name_to_id.get(name)

        if not champion_id:
            print(f"Skipping {name} (no Riot ID match)")
            continue

        splash_url = (
            f"https://ddragon.leagueoflegends.com/cdn/img/champion/loading/"
            f"{champion_id}_0.jpg"
        )

        champion_info.append({
            "name": name.lower().replace(" ", "").replace("'", ""),
            "faction": faction,
            "splash": splash_url
        })

    return champion_info



def embedder(path_to_img):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    all_heros = listdir(path_to_img)
    embeddings = []
    labels = []

    for i in range(0, len(all_heros), 5):
        batch = all_heros[i: i+5]
        images = []
        for hero in batch:
            image = Image.open(f"{path_to_img}/{hero}").convert("RGB")
            images.append(image)

            labels.append(hero[:-4])
    
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                batch_emp = outputs.pooler_output.cpu().numpy()
                for emb in batch_emp:
                    embeddings.append(emb)

    return embeddings, labels



def normalize_champion_name(name):
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s



def pca_display(embeddings, labels, color_vals, color_label='Color'):

    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame({
        "PC1": pca_embeddings[:, 0],
        "PC2": pca_embeddings[:, 1],
        "Hero": labels,
        color_label: color_vals
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_label,
        hover_name="Hero",
        title=f"PCA projection of image embeddings (colored by {color_label})",
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        },
    )
    fig.show()


def tsne_display(embeddings, image_labels, color_vals, color_label='Color'):
    tsne = TSNE(n_components=2,  perplexity=30)
    tsne_embeddings = tsne.fit_transform(embeddings)

    tsne_df = pd.DataFrame({
        "index": range(0, embeddings.shape[0]),
        "TSNE1": tsne_embeddings[:, 0],
        "TSNE2": tsne_embeddings[:, 1],
        "label": image_labels,
        color_label: color_vals,
    })

    fig = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color=color_label,
        hover_name="label",
        title=f"t-SNE projection of image embeddings (colored by {color_label})",
        labels={
            "TSNE1": "t-SNE dimension 1",
            "TSNE2": "t-SNE dimension 2",
        },
        hover_data=['index']
    )
    fig.show()


def umap_display(embeddings, labels, color_vals, color_label='Color'):
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
    )

    umap_embeddings = umap_reducer.fit_transform(embeddings)

    umap_df = pd.DataFrame({
        "index": range(0, embeddings.shape[0]),
        "UMAP1": umap_embeddings[:, 0],
        "UMAP2": umap_embeddings[:, 1],
        "label": labels,
        color_label: color_vals,
    })

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color=color_label,
        hover_name="label",
        title=f"UMAP projection of image embeddings (colored by {color_label})",
        labels={
            "UMAP1": "UMAP dimension 1",
            "UMAP2": "UMAP dimension 2",
        },
        hover_data=["index"],
    )

    fig.show()


def color_by_column(column, df_gender, labels):
    mapping = {}
    if 'Champion Name' in df_gender.columns and column in df_gender.columns:
        for _, row in df_gender.iterrows():
            key = normalize_champion_name(row['Champion Name'])
            mapping[key] = row[column]
            for alias, canon in alias_map.items():
                if canon == key:
                    mapping[alias] = row[column]

    column_list = []
    for hero_name in labels:
        norm = normalize_champion_name(hero_name)
        val = mapping.get(norm, "Unknown")
        column_list.append(val)

    return column_list


df_gender = pd.read_csv("League of Legends Gender and Sexuality Data - Sheet1.csv")
champion_info = fetch_champions_with_faction_and_splash()

embeddings, labels = embedder("heros")
embeddings = np.array(embeddings)

alias_map = {'monkeyking': 'wukong',}

name_to_faction = {champ['name']: champ['faction'].capitalize() for champ in champion_info}
factions = []
for hero_name in labels:
    norm = normalize_champion_name(hero_name)
    norm = alias_map.get(norm, norm)
    factions.append(name_to_faction.get(norm, "Runeterra"))




pca_display(embeddings, labels, factions, color_label='Faction')

pca_display(embeddings, labels, color_by_column("Gender", df_gender, labels), color_label='Gender')

tsne_display(embeddings, labels, color_by_column("Species", df_gender, labels), color_label='Species')

umap_display(embeddings, labels, color_by_column("Attractive?", df_gender, labels), color_label='Attractive?')

umap_display(embeddings, labels, color_by_column("Muscular?", df_gender, labels), color_label='Muscular?')

tsne_display(embeddings, labels, color_by_column("Primary Role", df_gender, labels), color_label='Primary Role')










def find_optimal_clusters(embeddings: NDArray, max_clusters: int = 10, random_state=42):
    """
    Determine the optimal number of clusters for embedding data using multiple methods:
    1. Elbow Method (inertia)
    2. Silhouette Score
    3. Davies-Bouldin Index

    Parameters:
    -----------
    embeddings : numpy.ndarray
        The embedding vectors to cluster, shape (n_samples, n_features)
    max_clusters : int, optional (default=10)
        Maximum number of clusters to try
    random_state : int, optional (default=42)
        Random seed for KMeans

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        A figure with subplots showing the evaluation metrics
    optimal_k : dict
        Dictionary with suggested optimal k values by different methods
    """
    # Ensure we have enough data points
    max_clusters = min(max_clusters, len(embeddings) - 1)

    # Initialize lists to store metrics
    range_n_clusters = list(range(2, max_clusters + 1))
    inertia_values = []
    silhouette_values = []
    davies_bouldin_values = []

    # Calculate metrics for each number of clusters
    for n_clusters in range_n_clusters:
        # Initialize and fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)

        # Calculate silhouette score (higher is better)
        if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters with points
            silhouette_values.append(silhouette_score(embeddings, cluster_labels))
        else:
            silhouette_values.append(0)

        # Calculate Davies-Bouldin index (lower is better)
        davies_bouldin_values.append(davies_bouldin_score(embeddings, cluster_labels))

    # Find optimal clusters using different methods
    optimal_k = {}

    inertia_diffs = np.diff(inertia_values)
    inertia_diffs2 = np.diff(inertia_diffs)
    optimal_k['elbow'] = range_n_clusters[np.argmin(inertia_diffs2) + 1]

    # Silhouette method (maximize)
    optimal_k['silhouette'] = range_n_clusters[np.argmax(silhouette_values)]

    # Davies-Bouldin method (minimize)
    optimal_k['davies_bouldin'] = range_n_clusters[np.argmin(davies_bouldin_values)]

    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Elbow Method (Inertia)",
            "Silhouette Score (higher is better)",
            "Davies-Bouldin Index (lower is better)"
        ),
        vertical_spacing=0.15
    )


    fig.add_trace(
        go.Scatter(
            x=range_n_clusters,
            y=inertia_values,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[optimal_k['elbow']],
            y=[inertia_values[range_n_clusters.index(optimal_k['elbow'])]
               if optimal_k['elbow'] in range_n_clusters else 0],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Optimal k (Elbow)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=range_n_clusters,
            y=silhouette_values,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[optimal_k['silhouette']],
            y=[silhouette_values[range_n_clusters.index(optimal_k['silhouette'])]],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Optimal k (Silhouette)'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=range_n_clusters,
            y=davies_bouldin_values,
            mode='lines+markers',
            name='Davies-Bouldin Index',
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[optimal_k['davies_bouldin']],
            y=[davies_bouldin_values[range_n_clusters.index(optimal_k['davies_bouldin'])]],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Optimal k (Davies-Bouldin)'
        ),
        row=3, col=1
    )

    fig.update_layout(
        title='Optimal Number of Clusters Evaluation',
        height=800,
        width=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    fig.update_xaxes(title_text="Number of Clusters")

    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=2, col=1)
    fig.update_yaxes(title_text="Davies-Bouldin Index", row=3, col=1)

    return fig, optimal_k


fig, optimal_k = find_optimal_clusters(embeddings, max_clusters=30)
fig.show()