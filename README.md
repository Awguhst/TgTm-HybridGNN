# HybridGNN Polymer Property Predictor

This project presents a machine learning framework for the **simultaneous prediction** of two fundamental thermal properties of polymers:

- **Glass Transition Temperature (Tg)**
- **Melting Temperature (Tm)**

given their **monomer SMILES representations**.

The core model leverages a **Hybrid Graph Neural Network (HybridGNN)** architecture that combines **graph-based molecular structure** with **physicochemical descriptors** for accurate, data-driven predictions.

---

## 🎥 Demo

![Streamlit app GIF](doc/demo.gif)

> *Visualization of the interactive Streamlit web app for polymer property prediction.*

---

## Motivation

Understanding and predicting **polymer thermal properties** is essential for designing new materials in fields such as **packaging**, **coatings**, **electronics**, and **aerospace**. However, traditional experimental methods tend to be **resource intensive** and **time consuming**.

This project facilitates polymer property prediction by integrating modern **graph-based deep learning** techniques with **chemical descriptors**, providing a scalable and data-driven alternative to conventional experimental approaches.

---

## Overview

The prediction model is based on a **HybridGNN** architecture that:

- Constructs a graph representation of the monomer
- Extracts RDKit-based descriptors (e.g., molecular weight, LogP, H-bond acceptors)
- Applies **Principal Component Analysis (PCA)** to reduce descriptor dimensionality
- Fuses the learned **graph embeddings** with **descriptor vectors**
- Performs **multi-target regression** to jointly predict Tg and Tm

The dataset consists of **1,564 polymer samples** collected from various **peer-reviewed polymer chemistry publications**. Each sample includes the monomer’s **SMILES representation**, along with experimentally measured values for **Glass Transition Temperature (Tg)** and **Melting Temperature (Tm)**.

An interactive **Streamlit-based interface** allows users to input SMILES strings and instantly visualize and predict polymer properties.

---

## Similarity Feature

In addition to property prediction, the app includes a **molecular similarity search** feature. Given a **user-input SMILES string**, the model computes a **Tanimoto similarity score** between the query molecule and all entries in the dataset, based on **ECFP (Morgan) fingerprints**. The most similar polymers are then displayed along with their known **Tg** and **Tm** values, offering valuable context and interpretability.

This feature allows users to:

- Quickly identify structurally similar polymers  
- Compare predicted vs. experimental thermal properties  
- Gain insights from analogous known compounds  

The similarity search enhances transparency and helps users **interpret predictions by example**, bridging the gap between data-driven modeling and chemical intuition.

---

## Model Architecture

The HybridGNN consists of a stack of:

- `Graph Isomorphism Network (GIN)`
- `Graph Attention Network (GAT)`
- `GraphConv`

These layers extract multi-scale structural information, which is then combined with compressed descriptors through fully connected layers.

The model achieves a **cross-validation R² score of 0.80 for Tg** and **0.70 for Tm**, indicating strong predictive performance across both thermal properties.

---

> ⚠️ **Note:** This is a demo project.  
> File paths assume a specific folder structure.  
> If you're trying to run it, you may need to move files into the same folder or adjust the paths accordingly.

---

## References

- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? *Proceedings of the International Conference on Learning Representations (ICLR)*.  
  [https://doi.org/10.48550/arXiv.1810.00826](https://doi.org/10.48550/arXiv.1810.00826)

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.  
  [https://doi.org/10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)

- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. *International Conference on Machine Learning (ICML)*.  
  [https://doi.org/10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)

- Feinberg, E. N., et al. (2018). PotentialNet for Molecular Property Prediction. *ACS Central Science*, 4(11), 1520–1530.  
  [https://doi.org/10.1021/acscentsci.8b00507](https://doi.org/10.1021/acscentsci.8b00507)

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.  
  [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

- RDKit: Open-source cheminformatics software. (2006).  
  Available at: [http://www.rdkit.org](http://www.rdkit.org)

- Jolliffe, I.T. (2002). *Principal Component Analysis*. Springer Series in Statistics.  
  ISBN: 978-0-387-95442-4
"
