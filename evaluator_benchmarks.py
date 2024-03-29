import os
import requests
import numpy as np
import pandas as pd
import rdkit.Chem.MolStandardize

from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from multiprocessing import Pool
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, rdMolDescriptors


def calc_fingerprints(mols, fp_type="mg", radius=2, bit_size=2048):
    if type(mols) == Chem.rdchem.Mol:
        mols = [mols]

    if fp_type == "mg":
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m,
                                                  radius,
                                                  bit_size,
                                                  useChirality=True)
            for m in mols
        ]

    elif fp_type == "rdk":
        fps = [AllChem.RDKFingerprint(m, fpSize=bit_size) for m in mols]

    elif fp_type == "tt":
        fps = [
            rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                m, nBits=bit_size) for m in mols
        ]

    elif fp_type == "ap":
        fps = [
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                m, nBits=bit_size, includeChirality=True) for m in mols
        ]

    else:
        raise Exception("Supported options for 'fp_type': 'mg'/'rdk'/'tt'/'ap")

    return fps

# def calc_fp_similarity(x):
#     """
#     fp_types: one or multi of mg(MorganFingerprint)、rdk(RDKFingerprint)、tt(TopologicalTorsionFingerprint)、ap(AtomPairFingerprint)
#     """
#     fp_types = ('mg', 'ap')
#     is_smi = True
#     scaffold = False
#     ref_m, prb_m = x["GD"], x["OCMR"]
#     if is_smi:
#         ref_m = Chem.MolFromSmiles(ref_m)
#         prb_m = Chem.MolFromSmiles(prb_m)
#     if ref_m is None or prb_m is None:
#         return 0
#     if scaffold:
#         ref_m = MurckoScaffold.GetScaffoldForMol(ref_m)
#         prb_m = MurckoScaffold.GetScaffoldForMol(ref_m)
#     simi_values = []
#     for fp_type in fp_types:
#         fps = calc_fingerprints([ref_m, prb_m],
#                                 fp_type=fp_type,
#                                 radius=2,
#                                 bit_size=2048)
#         simi_values.append(calc_matrix(fps, fps)[0, 1])
#     return np.mean(simi_values)

def calc_matrix(fps1, fps2):
    row_num = len(fps1)
    col_num = len(fps2)
    simi_matrix = np.eye(col_num, row_num)

    for i in range(col_num):
        for j in range(row_num):
            simi_matrix[i, j] = TanimotoSimilarity(fps2[i], fps1[j])
    return simi_matrix

def calc_fp_similarity(x, name):
    """
    refrence from ABCNet : https://github.com/zhang-xuan1314/ABC-Net/blob/main/src/cal_acc.py
    """
    is_smi = True
    ref_m, prb_m = x["GD"], x[name]
    if is_smi:
        ref_m_test = Chem.MolFromSmiles(ref_m)
        prb_m_test = Chem.MolFromSmiles(prb_m)
    if ref_m_test is None or prb_m_test is None:
        return 0

    try:
        smiles = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(ref_m)
        smiles_pred = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(prb_m)
    except:
        return 0
    mol1 = Chem.MolFromSmiles(smiles)
    mol2 = Chem.MolFromSmiles(smiles_pred)

    morganfps1 = AllChem.GetMorganFingerprint(mol1, 3)
    morganfps2 = AllChem.GetMorganFingerprint(mol2, 3)
    morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
    return morgan_tani

def inference(file_path):
    api = 'http://localhost:1234/infer'
    with requests.Session() as s:
        result = s.post(api,
                        files={
                            'file_upload': open(file_path, "rb")
                        },
                        data={
                            "molvec": True,
                            "osra": True,
                            "imago": True,
                            "ocmr": True,
                            "ocmr_det": True
                        }).json()
        return result


def norm_func(smile_str):
    try:
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile_str),
                                 isomericSmiles=True,
                                 canonical=True)
    except:
        smile = ""

    if "." in smile:
        smiles = smile.split(".")
        smile = smiles[np.argmax([len(_) for _ in smiles])]
    return smile

def eva_data(df_map):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_res_path = os.path.join(base_dir, "public_res.csv")
    global exist_list
    for item in tqdm(df_map.iterrows()):
        single_row = []
        item = dict(item[1])
        set_name = item["Group"]


        if set_name in ["CLEF", "JPO", "UOB", "USPTO"]:
            GD_smiles = item["Smiles"]
            img_path = os.path.join(base_dir, "Data", "public_data", set_name,
                                    "{}.png".format(item["Key"]))
            if item["Key"] in exist_list:
                print("跳过，存在{}".format(item["Key"]))
                continue
            res = inference(img_path)
            ocmr_is_true = (norm_func(GD_smiles) == norm_func(res["ocmr"]))
            # ------ OCMR
            single_row.append(item["Key"])
            single_row.append(os.path.basename(img_path))
            single_row.append(GD_smiles)
            single_row.append(res["ocmr"])
            single_row.append(set_name)
            single_row.append(ocmr_is_true)
            # ------ OSRA
            osra_is_true = (norm_func(GD_smiles) == norm_func(res["osra"]))
            single_row.append(res["osra"])
            single_row.append(osra_is_true)
            # ------ MolVec
            molvec_is_true = (norm_func(GD_smiles) == norm_func(res["molvec"]))
            single_row.append(res["molvec"])
            single_row.append(molvec_is_true)
            # ------ imago
            imago_is_true = (norm_func(GD_smiles) == norm_func(res["imago"]))
            single_row.append(res["imago"])
            single_row.append(imago_is_true)

            single_row = [single_row]

        else:
            continue
        df_row = pd.DataFrame(single_row)

        if not os.path.exists(csv_res_path):
            df_row.to_csv(csv_res_path,
                          header=[
                              "key", "Img_path", "GD", "OCMR", "Group",
                              "Is_OCMR_true","OSRA", "Is_OSRA_true", "MolVec", "Is_MolVec_true", "Imago", "Is_Imago_true"
                          ],
                          index=False,
                          mode='a')
        else:
            df_row.to_csv(csv_res_path, header=False, index=False, mode='a')

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_res_path = os.path.join(base_dir, "public_res.csv")
    if not os.path.exists(csv_res_path):
        smiles_map = os.path.join(base_dir, "Data", "public_data",
                                  "refence_smiles.csv")
        df_map = pd.read_csv(smiles_map)
        exist_list = []
        df_map = df_map[df_map["Group"].isin(["CLEF", "JPO", "UOB", "USPTO"])]
        df_list = [
            df_map[:3000], df_map[3000:6000], df_map[6000:9000], df_map[9000:]
            # df_map[:6000], df_map[6000:]
        ]
        po = Pool(4)
        po.map(eva_data, df_list)
        # eva_data(df_map)
    else:
        smiles_map = os.path.join(base_dir, "Data", "public_data",
                                  "refence_smiles.csv")
        df_map = pd.read_csv(smiles_map)
        exist_list = pd.read_csv(csv_res_path).key.to_list()
        df_map = df_map[df_map["Group"].isin(["CLEF", "JPO", "UOB", "USPTO"])]
        df_list = [
            df_map[:3000], df_map[3000:6000], df_map[6000:9000], df_map[9000:]
            # df_map[:6000], df_map[6000:]
        ]
        po = Pool(4)
        po.map(eva_data, df_list)
        # eva_data(df_map)


    df = pd.read_csv(csv_res_path)
    df = df.fillna(" ")

    tqdm.pandas(desc='apply')
    # similarity
    df["OCMR_TanimotoSimilarity"] = df.progress_apply(calc_fp_similarity,
                                             name="OCMR",
                                             axis=1)
    df["MolVec_TanimotoSimilarity"] = df.progress_apply(calc_fp_similarity,
                                               name="MolVec",
                                               axis=1)
    df["OSRA_TanimotoSimilarity"] = df.progress_apply(calc_fp_similarity,
                                             name="OSRA",
                                             axis=1)
    df["Imago_TanimotoSimilarity"] = df.progress_apply(calc_fp_similarity,
                                              name="Imago",
                                              axis=1)

    print(df.groupby("Group").mean())

    df_UOB = df[df["Group"] == "UOB"]
    df_UOB.reset_index(drop=True, inplace=True)
    df_CLEF = df[df["Group"] == "CLEF"]
    df_CLEF.reset_index(drop=True, inplace=True)
    df_JPO = df[df["Group"] == "JPO"]
    df_JPO.reset_index(drop=True, inplace=True)
    df_USPTO = df[df["Group"] == "USPTO"]
    df_USPTO.reset_index(drop=True, inplace=True)
    writer = pd.ExcelWriter(os.path.join(base_dir, "all_result_public.xlsx"))
    df_UOB.to_excel(writer, sheet_name="UOB")
    df_CLEF.to_excel(writer, sheet_name="CLEF")
    df_JPO.to_excel(writer, sheet_name="JPO")
    df_USPTO.to_excel(writer, sheet_name="USPTO")
    writer.save()
    print("reports saved in:{}".format(
        os.path.join(base_dir, "all_result_public.xlsx")))
