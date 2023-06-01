import os
import sys
import json
import torch
import zipfile
import io
import pandas as pd
import csv
import json

#データの読み込み
def load_data():
    with zipfile.ZipFile("../datasets/open2ch-dialogue-corpus/corpus.zip", "r") as zip_ref:
        zip_ref.extractall("")

    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_name = file.split(".")[0]
        file_name = pd.read_csv(directory+"/" + file, sep="\t", header=None)
        
 
    
        with zipfile.ZipFile("../datasets/open2ch-dialogue-corpus/corpus.zip", "r") as zip_file:
            csv_file = zip_file.namelist()[0]

            df = pd.read_csv(zip_file.open(csv_file), sep="\t", header=None)

            print(df.head())
    """
    return 0

def replace(filenames):

    #コマンドからreplace_br.pyを実行
    for file in filenames:
        os.system(f"python replace_br.py --input_file corpus/{file}.tsv --output_file corpus/{file}_replaced.tsv")
    
    return 0

def shape(filenames):
    for file in filenames:
        data = []
        with open(f"corpus/{file}_replaced.tsv", 'r') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                # 各行から最初の2つの要素を取得し、リストに追加します。
                data.append(row[:2])

            # リストをデータフレームに変換します。
            df = pd.DataFrame(data)
            df.columns = ["input", "target"]
            #print(df.head(10))
            #tsvファイルに書き出し
            df.to_csv(f"./corpus/{file}_shaped.tsv", sep="\t", index=False)
    
def labelings(filenames):
    for file in filenames:
        data =[]
        df = pd.read_csv(f"corpus/{file}_shaped.tsv", sep="\t")
        #print(df.head(10))
        #一行ずつ読み込んでラベル付け
        for i in range(len(df)):
            print(df.at[i,"input"])
            print(df.at[i,"target"])

            row_data = {}
            row_data["input_text"] = df.at[i,"input"]
            row_data["target_text"] = df.at[i,"target"]
            data.append(row_data)

        
        #list to json
        with open(f"corpus/{file}_input.json", 'w') as f:
            #ensure_ascii=Falss これ精度変わるかも
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        #list to tsv
        df.to_csv(f"corpus/{file}_input.tsv", sep="\t", index=False)
        

    return 0
if __name__ == '__main__':
    filenames = ["newsplus", "news4vip", "livejupiter"]

    #load_data()
    print("load_data() is done")
    replace(filenames)
    print("replace() is done")
    shape(filenames)
    print("shaping() is done")
    labelings(filenames)
    print("labelings() is done")


        
    
