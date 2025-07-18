import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
import torch
import joblib
import numpy as np
import pandas as pd
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

# === Load Model + Scalers ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridGNN(hidden_channels=128, descriptor_size=12, dropout_rate=0.6)
model.load_state_dict(torch.load("hybrid_gnn.pt", map_location=device))
model.eval().to(device)
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# === Load Training Data ===
@st.cache_data
def load_training_data():
    df = pd.read_csv("polymer_tg_tm.csv")

    # Normalize column name just once
    df.rename(columns={"SMILES": "smiles"}, inplace=True)

    df = df[df["smiles"].notna()]
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df["fingerprint"] = df["mol"].apply(
        lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) if m else None
    )
    return df.dropna(subset=["fingerprint"])

training_df = load_training_data()

# === App UI ===
st.title("Polymer Property Predictor")
st.markdown("Predict **Glass Transition (Tg)** and **Melting Temperature (Tm)** from a polymer SMILES using a **HybridGNN** model.")

with st.form("predict_form"):
    smiles_input = st.text_input("💡 Enter a polymer SMILES string:", "*CC(*)c1ccc(C(=O)C(F)(F)F)cc1")
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

            # Prepare graph
            graph = from_smiles(smiles_input)
            graph.x = graph.x.float().to(device)
            graph.edge_index = graph.edge_index.to(device)
            batch_index = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)

            # Predict
            with torch.no_grad():
                pred, _ = model(graph.x, graph.edge_index, batch_index, desc_tensor)

            tg, tm = pred[0].cpu().numpy()

            # Results
            st.subheader("📈 Predicted Properties")
            col1, col2 = st.columns(2)
            col1.metric(label="Glass Transition Temperature (Tg)", value=f"{tg:.2f} °C")
            col2.metric(label="Melting Temperature (Tm)", value=f"{tm:.2f} °C")

            st.success("✅ Prediction complete!")

            # === Similar Compounds Section ===
            st.markdown("---")
            st.markdown("### 🔍 Top 5 Similar Compounds from Training Set")

            # Compute input fingerprint
            input_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

            # Compute similarity
            def tanimoto(fp1, fp2):
                return DataStructs.TanimotoSimilarity(fp1, fp2)

            training_df["similarity"] = training_df["fingerprint"].apply(lambda fp: tanimoto(input_fp, fp))
            top_similar = training_df.sort_values(by="similarity", ascending=False).head(5)

            for i, row in top_similar.iterrows():
                sim_mol = row["mol"]
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(Draw.MolToImage(sim_mol, size=(200, 200)))
                with col2:
                    st.markdown(f"**SMILES**: `{row['smiles']}`")
                    st.markdown(f"**Similarity Score**: `{row['similarity']:.2f}`")
                    if "Tg" in row and "Tm" in row:
                        st.markdown(f"**Tg**: `{row['Tg']:.2f} °C`")
                        st.markdown(f"**Tm**: `{row['Tm']:.2f} °C`")
