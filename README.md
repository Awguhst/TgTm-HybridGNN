# 🧬 HybridGNN Polymer Property Predictor

This project presents a machine learning framework for the **simultaneous prediction** of two fundamental thermal properties of polymers:

- **Glass Transition Temperature (Tg)**
- **Melting Temperature (Tm)**

given their **monomer SMILES representations**.

The core model leverages a **Hybrid Graph Neural Network (HybridGNN)** architecture that combines **graph-based molecular structure** with **physicochemical descriptors** for accurate, data-driven predictions.

---

## 🎥 Demo

![Streamlit app GIF](doc/demo.gif)

---

## 🎯 Motivation

Understanding and predicting **polymer thermal properties** is essential for designing new materials in fields such as **packaging**, **coatings**, **electronics**, and **aerospace**. However, traditional experimental methods tend to be **resource-intensive** and **time-consuming**.

This project facilitates polymer property prediction by integrating modern **graph-based deep learning** techniques with **chemical descriptors**, providing a scalable and data-driven alternative to conventional experimental approaches.

---

## 🔬 Overview

The prediction model is based on a **HybridGNN** architecture that:

- Constructs a graph representation of the monomer using atom/bond information
- Extracts RDKit-based descriptors (e.g., molecular weight, LogP, H-bond acceptors)
- Applies **Principal Component Analysis (PCA)** to reduce descriptor dimensionality
- Fuses the learned **graph embeddings** with **descriptor vectors**
- Performs **multi-target regression** to jointly predict Tg and Tm

An interactive **Streamlit-based interface** allows users to input SMILES strings and instantly visualize and predict polymer properties.

---

## 🧠 Model Architecture

The HybridGNN consists of a stack of:

- `Graph Isomorphism Network (GIN)`
- `Graph Attention Network (GAT)`
- `GraphConv` layers

These layers extract multi-scale structural information, which is then combined with compressed descriptors through fully connected layers.

---
