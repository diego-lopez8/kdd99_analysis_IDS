import pandas as pd
import numpy as np

train_df = pd.read_csv("../data/raw/train1.csv", index_col = 0)

train_df.columns = train_df.columns.map(lambda x: x.replace(":", ""))

test_df = pd.read_csv("../data/raw/test1.csv", index_col = 0)

test_df.columns = test_df.columns.map(lambda x: x.replace(":", ""))

#The discrepancy between train and test unique value numbers is much greater for attack compared to label, so let us test on label and not attack.

#Therefore, let's modify the train and test sets to have only the intersection in registered label values.

train_test_intersect = np.intersect1d(train_df.labels.unique(), test_df.labels.unique())

modified_train_df = train_df[train_df.labels.isin(train_test_intersect)].drop(columns = "attack")

modified_train_df = pd.get_dummies(data = modified_train_df, columns = ["protocol_type", "service", "flag"], prefix = ["protocol_type", "service", "flag"])

modified_test_df = test_df[test_df.labels.isin(train_test_intersect)].drop(columns = "attack")

modified_test_df = pd.get_dummies(data = modified_test_df, columns = ["protocol_type", "service", "flag"], prefix = ["protocol_type", "service", "flag"])

# Some services (ICMP, Red_I, etc) are present in either dataset, but not both.
# create dummy column that placeholds the service
# In Production & with more compute power, it is common to one-hot encode all "well known" services (ports 0-1023)
for col in modified_train_df.columns:
    if col not in modified_test_df.columns:
        modified_test_df[col] = [0] * len(modified_test_df.index)

for col in modified_test_df.columns:
    if col not in modified_train_df.columns:
        modified_train_df[col] = [0] * len(modified_train_df.index)

modified_train_df.to_csv("../data/processed/traindata.csv")

modified_test_df.to_csv("../data/processed/testdata.csv")
