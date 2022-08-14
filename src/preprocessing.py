# This file performs basic preprocessing on the KDD Cup data downloaded from the website: 
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
# TODO: One hot encoding / label encoding of categorical features
# TODO: combine both preprocessing scripts
import pandas as pd
import numpy as np

attack_arr  = np.genfromtxt("../data/raw/training_attack_types.txt", dtype=str)
attack_dict = {}
for pair in attack_arr:
    attack_dict[pair[0]] = pair[1]

names                  = np.genfromtxt("../data/raw/kddcup.names.txt", dtype=str, encoding=None, skip_header=1)
cols                   = np.append(names.transpose()[0], "labels")
test_df                = pd.read_csv("../data/raw/corrected")
test_df.columns        = cols
train_10pct_df         = pd.read_csv("../data/raw/kddcup.data_10_percent")
train_10pct_df.columns = cols
train_10pct_df         = train_10pct_df.dropna()
test_df                = test_df.dropna()
train_10pct_df['labels'] = train_10pct_df['labels'].str.replace(".", "")
train_10pct_df['attack'] = train_10pct_df['labels'].replace(attack_dict)
test_df['labels'] = test_df['labels'].str.replace(".", "")
test_df['attack'] = test_df['labels'].replace(attack_dict)
test_df.head(10)

# export
test_df.to_csv("../data/raw/test1.csv")
train_10pct_df.to_csv("../data/raw/train1.csv")