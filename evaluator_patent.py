import os
import requests
import numpy as np
import pandas as pd
from rdkit import Chem


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
                                 canonical=True).replace("\\",
                                                         "").replace("/", "")
    except:
        smile = ""

    if "." in smile:
        smiles = smile.split(".")
        smile = smiles[np.argmax([len(_) for _ in smiles])]
    return smile


if __name__ == "__main__":
    # --------------------------设置路径--------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data", "Patent")
    img_dir = os.path.join(data_dir, "images")

    # -------------------------读取LABEL-------------------------
    df = pd.read_csv(os.path.join(data_dir, "smiles_map.csv"))
    dic = {
        "key": [],
        "Img_path": [],
        "GD": [],
        "OCMR": [],
        "Is_OCMR_true": [],
        "MolVec": [],
        "Is_MolVec_true": [],
        "OSRA": [],
        "Is_OSRA_true": [],
        "Imago": [],
        "Is_Imago_true": []
    }

    # ------------------遍历CSV,评估方法之间的差异------------------
    index = 0
    for item in df.iterrows():
        index += 1
        print("Now processing {} / {}".format(index, len(df)))
        item = dict(item[1])
        dimg_path = item["ImagePath"]
        img_path = os.path.join(img_dir, os.path.basename(dimg_path))
        # 请求服务获得结果
        res = inference(img_path)
        dic["key"].append(item["Key"])
        dic["GD"].append(item["Smiles"])
        dic["Img_path"].append(item["ImagePath"])
        try:
            dic["OCMR"].append(res["ocmr"])
            is_same1 = norm_func(res["ocmr"])
            is_same2 = norm_func(item["Smiles"])
            if is_same1 == is_same2:
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
            if is_same1 == is_same2:
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
            if is_same1 == is_same2:
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
            if is_same1 == is_same2:
                dic["Is_Imago_true"].append(1)
            else:
                dic["Is_Imago_true"].append(0)
        except:
            dic["Imago"].append("")
            dic["Is_Imago_true"].append(0)

    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(base_dir, "Patent_res.csv"))
    print(
        "------------------------------------------------------------------------"
    )
    print("Imago acc：{}".format(np.sum(df.Is_Imago_true)))
    print("Osra acc：{}".format(np.sum(df.Is_OSRA_true)))
    print("MolVec acc：{}".format(np.sum(df.Is_MolVec_true)))
    print("OCMR acc：{}".format(np.sum(df.Is_OCMR_true)))
