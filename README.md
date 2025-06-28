# A Graph-Based Learning Framework for Simultaneous Tg and Tm Prediction of Polymers Using Hybrid GNN and PCA Descriptor Fusion

This project implements a deep learning pipeline for the **simultaneous prediction** of the **glass transition temperature (Tg)** and **melting temperature (Tm)** of **polymers** based on their **monomer SMILES representations**.

The core of the model is a **hybrid Graph Neural Network (GNN)** architecture composed of `GIN → GAT → GraphConv` layers that learn hierarchical structure from polymer graphs. These **graph embeddings** are fused with **physicochemical descriptors** (e.g., molecular weight, LogP, rotatable bonds), which are computed via **RDKit** and compressed using **Principal Component Analysis (PCA)** to reduce redundancy and noise.

---

## Key Features

- **Hybrid GNN model**: Combines structural graph learning with attention and refinement layers
- **RDKit-based molecular descriptors** Capture relevant polymer chemical features from monomer units
- **PCA compression**: Dimensionality reduction and noise filtering for descriptor fusion with graph node features
- **Multi-target regression**: Joint prediction of **Tg** and **Tm** in a single model

## References

- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? *Proceedings of the International Conference on Learning Representations (ICLR)*.  
  [https://doi.org/10.48550/arXiv.1810.00826](https://doi.org/10.48550/arXiv.1810.00826)

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.  
  [https://doi.org/10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)

- RDKit: Open-source cheminformatics software. (2006).  
  Available at: [http://www.rdkit.org](http://www.rdkit.org)

- Jolliffe, I.T. (2002). *Principal Component Analysis*. Springer Series in Statistics.  
  ISBN: 978-0-387-95442-4
