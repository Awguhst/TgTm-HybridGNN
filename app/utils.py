from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from torch_geometric.utils import from_smiles as tg_from_smiles

def calculate_descriptors(mol):
    if mol is None:
        return torch.zeros((1, 25))

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.Chi0(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.MolMR(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.PEOE_VSA1(mol),
        Descriptors.PEOE_VSA2(mol),
        Descriptors.SMR_VSA1(mol),
        Descriptors.SMR_VSA2(mol),
        Descriptors.EState_VSA1(mol),
        Descriptors.EState_VSA2(mol),
        Descriptors.VSA_EState1(mol)
    ]

    return torch.tensor(descriptors, dtype=torch.float).view(1, -1)

def from_smiles(smile):
    return tg_from_smiles(smile)
