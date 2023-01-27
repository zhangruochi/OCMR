from rdkit import Chem
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transform SMILES to kekule form.')
    parser.add_argument('--src', type=str, help='the source SMILES you want to transform')

    args = parser.parse_args()

    src_smi = args.src
    m = Chem.MolFromSmiles(src_smi)
    Chem.Kekulize(m)
    print(Chem.MolToSmiles(m,kekuleSmiles=True))