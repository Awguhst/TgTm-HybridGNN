import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import joblib
import numpy as np
from rdkit.Chem import Descriptors
from hybrid_gnn import HybridGNN
from utils import calculate_descriptors, from_smiles

# === App Config ===
st.set_page_config(page_title="Polymer Tg/Tm Predictor", layout="centered", page_icon="🧪")

# === Custom CSS ===
st.markdown("""
    <style>
        /* Base background & text */
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #f1f1f1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Headings */
        h1, h2, h3 {
            color: #61dafb;
            font-weight: 700;
            letter-spacing: 1px;
        }

        /* Buttons */
        .stButton>button {
            color: white;
            background: linear-gradient(135deg, #1f77b4, #3a86ff);
            padding: 0.6rem 1.5rem;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #3a86ff, #1f77b4);
            cursor: pointer;
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            background-color: #1c1c1c;
            color: #f1f1f1;
            border: 1.5px solid #3a86ff;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            transition: border-color 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #61dafb;
            outline: none;
            box-shadow: 0 0 8px #61dafb;
        }

        /* Form container */
        .stForm {
            background-color: #1a1a1a;
            padding: 25px 30px;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(97, 218, 251, 0.15);
        }

        /* Metric boxes */
        .stMetric {
            background-color: #222222;
            color: #f1f1f1;
            border-radius: 12px;
            padding: 16px 20px;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: inset 0 0 5px rgba(97, 218, 251, 0.2);
        }

        /* Center molecule image */
        .css-1d391kg img {
            border-radius: 16px;
            box-shadow: 0 0 20px rgba(97, 218, 251, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# === Title ===
st.title("🧪 Polymer Property Predictor")
st.markdown("Predict **Glass Transition (Tg)** and **Melting Temperature (Tm)** from a polymer SMILES using a **HybridGNN** model.")

# === Load Model + Scalers ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridGNN(hidden_channels=128, descriptor_size=12, dropout_rate=0.6)
model.load_state_dict(torch.load("hybrid_gnn.pt", map_location=device))
model.eval().to(device)

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# === Input Form ===
with st.form("predict_form"):
    smiles_input = st.text_input("💡 Enter a polymer SMILES string:", "*CC(*)C(C)C")
    submit = st.form_submit_button("🔍 Predict")

if submit:
    with st.spinner("🔬 Analyzing molecule..."):
        mol = Chem.MolFromSmiles(smiles_input)

        if mol is None:
            st.error("❌ Invalid SMILES string. Please try again.")
        else:
            # Show molecule
            mol_img = Draw.MolToImage(mol, size=(700, 700))
            st.markdown("### 🧬 Input Molecule")
            col = st.columns([1, 1])[0]  # center image
            with col:
                st.image(mol_img, use_container_width=True)

            # Compute descriptors
            descriptors = calculate_descriptors(mol)
            desc_scaled = scaler.transform(descriptors)
            desc_pca = pca.transform(desc_scaled)
            desc_tensor = torch.tensor(desc_pca, dtype=torch.float).to(device)

            # Graph input
            graph = from_smiles(smiles_input)
            graph.x = graph.x.float().to(device)
            graph.edge_index = graph.edge_index.to(device)  # ✅ fix: move to device
            batch_index = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)

            # Inference
            with torch.no_grad():
                pred, _ = model(graph.x, graph.edge_index, batch_index, desc_tensor)

            tg, tm = pred[0].cpu().numpy()

            # Results
            st.subheader("📈 Predicted Properties")
            col1, col2 = st.columns(2)
            col1.metric(label="Glass Transition Temperature (Tg)", value=f"{tg:.2f} °C")
            col2.metric(label="Melting Temperature (Tm)", value=f"{tm:.2f} °C")

            st.success("✅ Prediction complete!")