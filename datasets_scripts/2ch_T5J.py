import os
import sys
import json
import torch
import zipfile
import io
import pandas as pd
import csv

#データの読み込み
def load_data():
    directory = ""
    with zipfile.ZipFile("../datasets/open2ch-dialogue-corpus/corpus.zip", "r") as zip_ref:
        zip_ref.extractall("")

    directory = "corpus"
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

def replace():

    #コマンドからreplace_br.pyを実行
    os.system("python replace_br.py --input_file corpus/newsplus.tsv --output_file corpus/newsplus_replaced.tsv")
    return 0

def shaping():
    #df = pd.read_csv("corpus/newsplus_replaced.tsv", sep="\t", header=None)
    #df = df.iloc[:, :2]
    data = []

    # ファイルを開きます。
    with open("corpus/newsplus_replaced.tsv", 'r') as file:
        # csv.readerを使用してファイルを読み込みます。
        reader = csv.reader(file, delimiter='\t')

        # ファイル内の各行を反復処理します。
        for row in reader:
            # 各行から最初の2つの要素を取得し、リストに追加します。
            data.append(row[:2])

    # リストをデータフレームに変換します。
    df = pd.DataFrame(data)
    #df.columns = ["input", "target"]
    #print(data.head(10))
    #tsvファイルに書き出し
    df.to_csv("corpus/newsplus_shaped.tsv", sep="\t", index=False)
    


if __name__ == '__main__':
    load_data()
    print("load_data() is done")
    replace()
    print("replace() is done")
    shaping()
    print("shaping() is done")


        
    
