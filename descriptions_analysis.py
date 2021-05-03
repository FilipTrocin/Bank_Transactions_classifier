import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from common_methods import remove_numbers, remove_pymt_types, remove_extra_symbols

features = pd.read_csv("data/bank_transaction_features.csv", sep=",")
labels = pd.read_csv("data/bank_transaction_labels.csv", sep=",")

feat_headers = ["bank_transaction_id", "bank_transaction_description",
                "bank_transaction_amount", "bank_transaction_type"]
lab_headers = ["bank_transaction_id", "bank_transaction_category", "bank_transaction_dataset"]

all_desc = np.array(features["bank_transaction_description"]).astype(str)

# modified all_desc, changed format
desc_cleaned = remove_extra_symbols(remove_pymt_types(remove_numbers(all_desc)))


# remove empty records
while '' in desc_cleaned:
    desc_cleaned.remove('')


def take_first_few(n, iterable):
    return list(islice(iterable, n))


unique, counts = np.unique(desc_cleaned, return_counts=True)
most_repeatable = take_first_few(10, sorted(zip(unique, counts), key=lambda x: x[1], reverse=True))
least_repeatable = take_first_few(9, sorted(zip(unique, counts), key=lambda x: x[1]))

most_rep_desc_to_list = [x[0] for x in most_repeatable]
most_rep_val_to_list = [x[1] for x in most_repeatable]

least_rep_desc_to_list = [x[0] for x in least_repeatable]
least_rep_val_to_list = [x[1] for x in least_repeatable]


def print_barh(x_axis, y_axis, label, title):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(y_axis))
    x_pos = np.array(x_axis)

    ax.barh(y_pos, x_pos, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_axis)
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_title(title)

    plt.show()


x_axis_label = 'Number of Transactions'

print_barh(most_rep_val_to_list, most_rep_desc_to_list, x_axis_label, 'The most common descriptions')
print_barh(least_rep_val_to_list, least_rep_desc_to_list, x_axis_label, 'The least common descriptions')
