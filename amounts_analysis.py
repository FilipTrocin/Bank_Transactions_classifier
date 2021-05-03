import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

features = pd.read_csv("data/bank_transaction_features.csv", sep=",")
labels = pd.read_csv("data/bank_transaction_labels.csv", sep=",")

feat_headers = ["bank_transaction_id", "bank_transaction_description",
                "bank_transaction_amount", "bank_transaction_type"]
lab_headers = ["bank_transaction_id", "bank_transaction_category", "bank_transaction_dataset"]


all_desc = np.array(features["bank_transaction_amount"]).astype(str)
modified = [x.replace("-", "") for x in all_desc]
to_float = sorted([float(x) for x in modified])

cnt = Counter()

for val in to_float:
    binn = val//10
    cnt[binn] += 1

nums_to_list = [x[1] for x in cnt.items()]

pie_labels = np.char.array(["£0 - £10", "£10 - £20", "£20 - £30", "£30 - £40", "£40 - £50", "£50 - £60", "£60 - £70",
                            "£70 - £80", "£80 - £90", " > £100"])
pie_values = np.array(nums_to_list)
percents = 100.*pie_values/pie_values.sum()

patches, texts = plt.pie(pie_values, shadow=True)
legend = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(pie_labels, percents)]

sort_legend = False
if sort_legend:
    patches, legend, dummy = zip(*sorted(zip(patches, legend, pie_values), key=lambda x: x[2], reverse=True))

plt.legend(patches, legend, loc='center left', bbox_to_anchor=(-0.4, 0.9), fontsize=8)
plt.title("Number of transactions and amounts\n percentage-wise", fontweight="bold")
plt.show()
