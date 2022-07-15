import requests
import os, time, random
from rdkit import Chem
from multiprocessing import Pool


def inference(file):
    api = 'http://192.168.1.221:1234/infer'
    file = os.path.abspath(file)
    # 获取上传文件
    with requests.Session() as s:
        result = s.post(api,
                        files={
                            'file_upload': open(file, "rb")
                        },
                        data={
                            "molvec": True,
                            "osra": True,
                            "imago": True,
                            "ocmr": True
                        }).json()
        return result


if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, "test.png")
    res = inference(file)
    print(res)