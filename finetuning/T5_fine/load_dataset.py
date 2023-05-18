import zipfile

# ZIPファイルを解凍する
with zipfile.ZipFile("datasets/open2ch-dialogue-corpus/corpus.zip", "r") as zip_ref:
    zip_ref.extractall("T5_fine")

# 解凍したファイルを読み込む
with open('target_directory/your_file.txt', 'r') as f:
    content = f.read()
    print(content)
