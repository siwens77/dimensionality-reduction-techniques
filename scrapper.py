import scrapy
import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests 
import re
from transformers import pipeline,  AutoImageProcessor, AutoModel
import os
from os import listdir
from PIL import Image
import torch
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import umap.umap_ as umap


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



def embedder2(path_to_img):
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

 
embeddings, labels = embedder2("heros")
embeddings = np.array(embeddings)

def pca_display(embeddings, labels):

    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)


    pca_df = pd.DataFrame({
        "PC1": pca_embeddings[:, 0],
        "PC2": pca_embeddings[:, 1],
        "Hero": labels
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Hero",
        hover_name="Hero",
        title="PCA projection of image embeddings",
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        },
    )
    fig.show()


pca_display(embeddings, labels)



def tsne_display(embeddings, image_labels):
    tsne = TSNE(n_components=2,  perplexity=30) # random_state=42,
    tsne_embeddings = tsne.fit_transform(embeddings)

    tsne_df = pd.DataFrame({
        "index": range(0, embeddings.shape[0]),
        "TSNE1": tsne_embeddings[:, 0],
        "TSNE2": tsne_embeddings[:, 1],
        "label": image_labels,
    })

    fig = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="label",
        hover_name="label",
        title="t-SNE projection of image embeddings",
        labels={
            "TSNE1": "t-SNE dimension 1",
            "TSNE2": "t-SNE dimension 2",
        },
        hover_data=['index']
    )
    fig.show()

tsne_display(embeddings, labels)

def umap_display(embeddings, labels):
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
    })

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="label",
        hover_name="label",
        title="UMAP projection of image embeddings",
        labels={
            "UMAP1": "UMAP dimension 1",
            "UMAP2": "UMAP dimension 2",
        },
        hover_data=["index"],
    )

    fig.show()

umap_display(embeddings, labels)