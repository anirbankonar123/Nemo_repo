import wget
import pandas as pd
import os
DATA_DIR = '/home/anish/Downloads'

# wget.download('https://dl.fbaipublicfiles.com/glue/data/MNLI.zip', DATA_DIR)

num_examples = 5
df = pd.read_csv(os.path.join(DATA_DIR, "MNLI", "dev_matched.tsv"), sep="\t")
# for sent1, sent2, label in zip(df['sentence1'].tolist(), df['sentence2'].tolist(), df['gold_label'].tolist()):
#     print("sentence 1: ", sent1)
#     print("sentence 2: ", sent2)
#     print("label: ", label)
#     print("===================")
print("validaton set length:"+str(len(df)))

import csv
reader = csv.reader(open(os.path.join(DATA_DIR, "MNLI", "train.tsv")))
no_lines= len(list(reader))
print(no_lines)

