# A Graph-Based Learning Framework for Simultaneous Tg and Tm Prediction of Polymers Using Hybrid GNN and PCA Descriptor Fusion

This project introduces a deep learning pipeline for the **simultaneous prediction** of **glass transition temperature (Tg)** and **melting temperature (Tm)** in **polymeric materials**, based on their SMILES representations.

The core of the model is a **hybrid Graph Neural Network (GNN)** architecture composed of `GIN → GAT → GraphConv` layers that learn hierarchical structure from polymer graphs. These **graph embeddings** are fused with **physicochemical descriptors** (e.g., molecular weight, LogP, rotatable bonds), which are compressed using **Principal Component Analysis (PCA)** to reduce redundancy and noise.

---

## Key Features

- **Hybrid GNN model**: Combines structural graph learning with attention and refinement layers (`GIN → GAT → GraphConv`)
- **RDKit-based molecular descriptors** for polymer-specific chemical information
- **PCA-based fusion**: Dimensionality reduction and noise filtering of descriptor vectors
- **Multi-target regression**: Joint prediction of **Tg** and **Tm** in a single model
- **Optional SMILES augmentation**: Randomized SMILES for better generalization on limited datasets
