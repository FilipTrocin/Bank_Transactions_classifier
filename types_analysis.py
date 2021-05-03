import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

features = pd.read_csv("data/bank_transaction_features.csv", sep=",")
labels = pd.read_csv("data/bank_transaction_labels.csv", sep=",")

feat_headers = ["bank_transaction_id", "bank_transaction_description",
                "bank_transaction_amount", "bank_transaction_type"]
lab_headers = ["bank_transaction_id", "bank_transaction_category", "bank_transaction_dataset"]


all_types = np.array(features["bank_transaction_type"]).astype(str)

d = Counter()

for x in all_types:
    d[x] += 1

sorted_types = sorted(d.items(), key=lambda x: x[1], reverse=True)
print(sorted_types)
values_to_list = [x[1] for x in d.items()]
desc_to_list = [x[0] for x in d.items()]

plt.pie(values_to_list, labels=desc_to_list, autopct='%1.1f%%', shadow=True)
plt.title("Number of transactions and types\n percentage-wise", fontweight="bold")
plt.show()
