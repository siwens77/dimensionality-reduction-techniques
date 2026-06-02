# Dimensionality reduction techniques

## Overview

A data pipeline that aggregates League of Legends champion artwork, extracts semantic image features using Meta's **DINOv2 Vision Transformer**, and visualizes stylistic similarities using various dimensionality reduction techniques. Supports coloring by multiple champion attributes: faction, gender, species, role, and more.

## 1. Data Collection (APIs)

Because game data is scattered, this project unifies three different sources to build a complete dataset:

* **CommunityDragon:** Scrapes raw, high-resolution base `.jpg` loading screens directly from un-packed game client files.
* **Data Dragon:** Riot's official site used to fetch foundational champion IDs and metadata.
* **Riot Universe API:** An undocumented JSON endpoint used to programmatically map champions to their lore-accurate factions.
* **Gender & Sexuality CSV:** External data source for champion demographics (gender, species, attractiveness, etc.).

## 2. Dimensionality Reduction

DINOv2 outputs hundreds of dimensions for a single image. To visualize how champion art styles cluster, the pipeline compresses these dense vectors into 2D space using three algorithms:

* **PCA (Principal Component Analysis):** The linear baseline. It finds the axes of greatest variance, preserving the global structure but often resulting in overlapping data points.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique that strictly prioritizes local similarities. It is very good at grouping visually identical art together but distorts macro-distances.
* **UMAP (Uniform Manifold Approximation and Projection):** The modern standard. It relies on Riemannian geometry to balance tight local clustering while preserving the accurate global distances between distinct clusters.

## 3. Visualization & Attribute Coloring

All three dimensionality reduction plots can be colored by champion attributes:

* **Faction:** In-game allegiances (Demacia, Noxus, Piltover, etc.)
* **Gender:** M/F/Unknown classification from the CSV
* **Species:** Champion species type (Humanoid, Monster, etc.)
* **Attractive?:** Visual appeal rating from the CSV
* **Muscular?:** Physique characteristics from the CSV
* **Primary Role:** Champion's main gameplay role (Fighter, Mage, Tank, etc.)

This allows exploration of how different champion attributes correlate with visual embedding space.

## Acknowledgements

* Champion data and images are sourced from [CommunityDragon](https://communitydragon.org/), [Data Dragon](https://developer.riotgames.com/docs/lol), and the Riot Universe API.
* Gender/species data from League of Legends Gender and Sexuality dataset.
* Image embeddings are powered by Meta's [DINOv2](https://huggingface.co/facebook/dinov2-base).
