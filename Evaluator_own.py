import os
import requests
import numpy as np
import pandas as pd
import Levenshtein as le

from rdkit import Chem
from multiprocessing import Pool
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, rdMolDescriptors

def calc_fingerprints(mols, fp_type="mg", radius=2, bit_size=2048):
    if type(mols) == Chem.rdchem.Mol:
        mols = [mols]

    if fp_type=="mg":
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, bit_size, useChirality=True) for m in mols]

    elif fp_type=="rdk":
        fps = [AllChem.RDKFingerprint(m, fpSize=bit_size) for m in mols]

    elif fp_type=="tt":
        fps = [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=bit_size) for m in mols]

    elif fp_type == "ap":
        fps = [rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=bit_size, includeChirality=True) for m in mols]

    else:
        raise Exception("Supported options for 'fp_type': 'mg'/'rdk'/'tt'/'ap")

    return fps

def calc_matrix(fps1, fps2):
    row_num = len(fps1)
    col_num = len(fps2)
    simi_matrix = np.eye(col_num,row_num)

    for i in range(col_num):
        for j in range(row_num):
            simi_matrix[i,j] = TanimotoSimilarity(fps2[i],fps1[j])
    return simi_matrix

def calc_fp_similarity(x,name):
    """
    fp_types: one or multi of mg(MorganFingerprint)、rdk(RDKFingerprint)、tt(TopologicalTorsionFingerprint)、ap(AtomPairFingerprint)
    """
    fp_types=('mg', 'ap')
    is_smi=True
    scaffold=False
    ref_m, prb_m = x["GD"],x[name]
    if is_smi:
        ref_m = Chem.MolFromSmiles(ref_m)
        prb_m = Chem.MolFromSmiles(prb_m)
    if ref_m is None or prb_m is None:
        return 0
    if scaffold:
        ref_m = MurckoScaffold.GetScaffoldForMol(ref_m)
        prb_m = MurckoScaffold.GetScaffoldForMol(ref_m)
    simi_values = []
    for fp_type in fp_types:
        fps = calc_fingerprints([ref_m, prb_m], fp_type=fp_type, radius=2, bit_size=2048)
        simi_values.append(calc_matrix(fps, fps)[0, 1])
    return np.mean(simi_values)


def inference(file_path):
    api='http://localhost:1234/infer'
    # 创建保存路径
    # 获取上传文件
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
                                 canonical=True).replace("\\",
                                                         "").replace("/", "")
    except:
        smile = ""

    if "." in smile:
        smiles = smile.split(".")
        smile = smiles[np.argmax([len(_) for _ in smiles])]
    return smile

def edit_distance(x, name):
    sm1 = norm_func(x["GD"])
    sm2 = norm_func(x[name])
    return le.distance(sm1,sm2)

if __name__ == "__main__":
    # --------------------------设置路径--------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data", "Patent")
    img_dir = os.path.join(data_dir, "images")
    
    # -------------------------读取LABEL-------------------------
    df = pd.read_csv(os.path.join(data_dir, "smiles_map.csv"))
    dic={"key": [], "Img_path": [], "GD": [], 
            "OCMR": [], "Is_OCMR_true": [],
            "MolVec": [], "Is_MolVec_true": [],
            "OSRA": [], "Is_OSRA_true": [],
            "Imago": [], "Is_Imago_true": []}

    # ------------------遍历CSV,评估方法之间的差异------------------
    index=0
    for item in df.iterrows():
        index+=1
        print("Now processing {} / {}".format(index,len(df)))
        item = dict(item[1])
        dimg_path = item["ImagePath"]
        img_path = os.path.join(img_dir,os.path.basename(dimg_path))
        # 请求服务获得结果
        res = inference(img_path)
        dic["key"].append(item["Key"])
        dic["GD"].append(item["Smiles"])
        dic["Img_path"].append(item["ImagePath"])
        try:
            dic["OCMR"].append(res["ocmr"])
            is_same1 = norm_func(res["ocmr"])
            is_same2 = norm_func(item["Smiles"])
            if is_same1==is_same2:
                dic["Is_OCMR_true"].append(1)
            else:
                dic["Is_OCMR_true"].append(0)
        except:
            dic["OCMR"].append("")
            dic["Is_OCMR_true"].append(0)
        try:
            dic["MolVec"].append(res["molvec"])
            is_same1 = norm_func(res["molvec"])
            is_same2 = norm_func(item["Smiles"])
            if is_same1==is_same2:
                dic["Is_MolVec_true"].append(1)
            else:
                dic["Is_MolVec_true"].append(0)
        except:
            dic["MolVec"].append("")
            dic["Is_MolVec_true"].append(0)
        try:
            dic["OSRA"].append(res["osra"])
            is_same1 = norm_func(res["osra"])
            is_same2 = norm_func(item["Smiles"])
            if is_same1==is_same2:
                dic["Is_OSRA_true"].append(1)
            else:
                dic["Is_OSRA_true"].append(0)
        except:
            dic["OSRA"].append("")
            dic["Is_OSRA_true"].append(0)
        try:
            dic["Imago"].append(res["imago"])
            is_same1 = norm_func(res["imago"])
            is_same2 = norm_func(item["Smiles"])
            if is_same1==is_same2:
                dic["Is_Imago_true"].append(1)
            else:
                dic["Is_Imago_true"].append(0)
        except:
            dic["Imago"].append("")
            dic["Is_Imago_true"].append(0)
    
    df = pd.DataFrame(dic)
    df = df.fillna("c")
    # 相似性
    df["OCMR_TanimotoSimilarity"] = df.apply(calc_fp_similarity, name="OCMR", axis=1)
    df["MolVec_TanimotoSimilarity"] = df.apply(calc_fp_similarity, name="MolVec", axis=1)
    df["OSRA_TanimotoSimilarity"] = df.apply(calc_fp_similarity, name="OSRA", axis=1)
    df["Imago_TanimotoSimilarity"] = df.apply(calc_fp_similarity, name="Imago", axis=1)
    # 编辑距离
    df["OCMR_edit_distance"] = df.apply(edit_distance, name="OCMR", axis=1)
    df["MolVec_edit_distance"] = df.apply(edit_distance, name="MolVec", axis=1)
    df["OSRA_edit_distance"] = df.apply(edit_distance, name="OSRA", axis=1)
    df["Imago_edit_distance"] = df.apply(edit_distance, name="Imago", axis=1)
    # 打印结果
    print(df.mean())
    df.to_csv(os.path.join(base_dir,"Patent_res.csv"))
