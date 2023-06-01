import pandas as pd

# ファイル名をリストに保存
filenames = ["corpus/livejupiter_input.tsv", "corpus/news4vip_input.tsv", "corpus/newsplus_input.tsv"]

# 各ファイルをヘッダーなしでDataFrameとして読み込み、リストに追加
df_list = [pd.read_csv(filename, sep='\t', header=None) for filename in filenames]

# 全てのDataFrameを結合
df_combined = pd.concat(df_list, ignore_index=True)

# 結果を新しいTSVファイルに保存
df_combined.to_csv('corpus/input.tsv', sep='\t', index=False, header=False)
