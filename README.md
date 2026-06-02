# Dimensionality reduction techniques

## Overview

A data pipeline that aggregates League of Legends champion artwork, extracts semantic image features using Meta's **DINOv2 Vision Transformer**, and visualizes stylistic similarities across in-game factions using various dimensionality reduction techniques.

## 1. Data Collection (APIs)

Because game data is scattered, this project unifies three different sources to build a complete dataset:

* **CommunityDragon:** Scrapes raw, high-resolution base `.jpg` loading screens directly from un-packed game client files.
* **Data Dragon:** Riot's official site used to fetch foundational champion IDs and metadata.
* **Riot Universe API:** An undocumented JSON endpoint used to programmatically map champions to their lore-accurate factions.

## 2. Dimensionality Reduction

DINOv2 outputs hundreds of dimensions for a single image. To visualize how champion art styles cluster by faction, the pipeline compresses these dense vectors into 2D space using three algorithms:

* **PCA (Principal Component Analysis):** The linear baseline. It finds the axes of greatest variance, preserving the global structure but often resulting in overlapping data points.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique that strictly prioritizes local similarities. It is very good at grouping visually identical art together but distorts macro-distances.
* **UMAP (Uniform Manifold Approximation and Projection):** The modern standard. It relies on Riemannian geometry to balance tight local clustering while preserving the accurate global distances between distinct clusters.


## Acknowledgements

* Champion data and images are sourced from [CommunityDragon](https://communitydragon.org/), [Data Dragon](https://developer.riotgames.com/docs/lol), and the Riot Universe API.
* Image embeddings are powered by Meta's [DINOv2](https://huggingface.co/facebook/dinov2-base).
