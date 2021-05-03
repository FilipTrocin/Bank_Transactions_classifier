from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from collections import Counter
from common_methods import remove_numbers, remove_pymt_types, remove_extra_symbols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import nltk
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
nltk.download("stopwords")


features = pd.read_csv("data/bank_transaction_features.csv", sep=",")
feat_headers = ["bank_transaction_id", "bank_transaction_description",
                "bank_transaction_amount", "bank_transaction_type"]

lab_headers = ["bank_transaction_id", "bank_transaction_category", "bank_transaction_dataset"]

train_l = []  # 10_000 training labels (80% of data)
validate_l = []  # 2500 validation labels (20% of data)

with open("data/bank_transaction_labels.csv", mode='r') as f:
    reader = csv.reader(f, delimiter=',')
    for entry in reader:
        if entry[2] == "TRAIN":
            train_l.append(entry)
        if entry[2] == "VAL":
            validate_l.append(entry)

train_f = []  # 10_000 training features
validate_f = []  # 2500 validation features

train_no = len(train_l)
validate_no = len(validate_l)

with open("data/bank_transaction_features.csv", mode='r') as f_file:
    reader = csv.reader(f_file, delimiter=',')
    for entry in reader:
        if entry[0] != 'bank_transaction_id':
            if train_no != 0:
                train_f.append(entry)
                train_no = train_no - 1
            else:
                validate_f.append(entry)
                validate_no = validate_no - 1

# Select descriptions only from lists
train_f_arr = np.char.array([x[1] for x in train_f]).astype(str)
train_l_arr = np.char.array([x[1] for x in train_l]).astype(str)
validate_f_arr = np.char.array([x[1] for x in validate_f]).astype(str)
validate_l_arr = np.char.array([x[1] for x in validate_l]).astype(str)


num_unique_labels = len(np.unique(train_l_arr))


def remove_stopwords(array):
    modified = []
    stop = set(stopwords.words("english"))
    for entry in array:
        removed = [word for word in entry.split() if word not in stop]
        joined = " ".join(removed)
        modified.append(joined)
    return modified


# Formatting features using `remove_numbers`, `remove_pymt_types`, `remove_extra_symbols`, `remove_stopwords` methods
train_f_arr = np.char.array(remove_stopwords(remove_extra_symbols(remove_pymt_types(remove_numbers(train_f_arr)))))
validate_f_arr = np.char.array(remove_stopwords(remove_extra_symbols(remove_pymt_types(remove_numbers(validate_f_arr)))))


def count_words(array):
    cnt = Counter()
    for entry in array:
        for word in entry.split():
            cnt[word] += 1
    return cnt


unique_words = count_words(np.concatenate((train_f_arr, validate_f_arr)))
num_unique_words = len(unique_words)


# Turning each word into sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_f_arr)

f_index = tokenizer.word_index

train_f_sequences = tokenizer.texts_to_sequences(train_f_arr)
validate_f_sequences = tokenizer.texts_to_sequences(validate_f_arr)


l_tokenizer = Tokenizer(lower=False, split=' ')
l_tokenizer.fit_on_texts(train_l_arr)

l_index = l_tokenizer.word_index

train_l_sequences = l_tokenizer.texts_to_sequences(train_l_arr)
validate_l_sequences = l_tokenizer.texts_to_sequences(validate_l_arr)

# how many words at in each entry
max_length = 5

# Making each entry the same length equal to max_length
train_f_padded = pad_sequences(train_f_sequences, maxlen=max_length, padding="post", truncating="post")
validate_f_padded = pad_sequences(validate_f_sequences, maxlen=max_length, padding="post", truncating="post")
train_l_padded = pad_sequences(train_l_sequences, maxlen=max_length, padding="post", truncating="post")
validate_l_padded = pad_sequences(validate_l_sequences, maxlen=max_length, padding="post", truncating="post")

reversed_f_index = dict([(value, key) for (key, value) in f_index.items()])
reversed_l_index = dict([(value, key) for (key, value) in l_index.items()])

try:
    model = keras.models.load_model('transactions_model.h5')
    model.summary()
except OSError:
    model = keras.models.Sequential()
    model.add(layers.Embedding(num_unique_words, 64, input_length=max_length))
    model.add(layers.SpatialDropout1D(0.2))
    model.add(layers.LSTM(20, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(num_unique_labels, activation='relu'))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'], learning_rate=0.001)
    history = model.fit(train_f_padded, train_l_padded, epochs=12,
                        validation_data=(validate_f_padded, validate_l_padded), batch_size=32)
    model.save("transactions_model.h5")

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


loss, acc = model.evaluate(validate_f_padded, validate_l_padded)
print("Loss value: {:.3f} ".format(loss), "Accuracy: {:5.2f}%\n".format(100 * acc))

text_index = 32
text = validate_f_padded[text_index]

pred = model.predict(np.expand_dims(text, axis=0))[0]
for cl in range(num_unique_labels):
    print("Probability for class {}: {}".format(cl, pred[cl]))
print("\nThe prediction class is {}".format(np.argmax(pred)))
print("The correct class is {}\n".format(np.argmax(validate_l_padded[text_index])))

