# -*- coding: utf-8 -*-

# Εισαγωγή των απαραίτητων βιβλιοθηκών
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import re
import string
import time
import joblib

data = pd.read_csv('train.csv')
# Εκτύπωση των πρώτων 5 γραμμών για κάθε dataset για επιβεβαίωση
print("Dataset:")
print(data.head())

data.columns

df = data.drop(["title", "author","id"], axis = 1)
df = df.dropna()
df = df.reset_index(drop=True)
df.head()

df.isnull().sum()

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)
# Εκτύπωση των πρώτων 5 γραμμών και
print("Εκτύπωση των πρώτων 5 γραμμών df")
print(df.head())

df.shape

# Χωρισμός των δεδομένων σε σύνολα εκπαίδευσης και ελέγχο
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# TF-IDF Vectorizer για Logistic Regression και SVM
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Αποθήκευση TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Εκπαίδευση Logistic Regression
print("Εκπαίδευση Logistic Regression...")
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train_tfidf, y_train)
# Αποθήκευση εκπαιδευμένου μοντέλου
joblib.dump(logistic_regression_model, 'logistic_regression_model.joblib')


# Εκπαίδευση SVM
print("Εκπαίδευση SVM...")
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
# Αποθήκευση εκπαιδευμένου μοντέλου
joblib.dump(svm_model, 'svm_model.joblib')

# Εκπαίδευση Νευρωνικού Δικτύου
print("Εκπαίδευση Νευρωνικού Δικτύου")
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_nn = tokenizer.texts_to_sequences(X_train)
X_test_nn = tokenizer.texts_to_sequences(X_test)

joblib.dump(tokenizer, 'tokenizer.joblib')

X_train_nn = pad_sequences(X_train_nn, maxlen=100, padding='post', truncating='post')
X_test_nn = pad_sequences(X_test_nn, maxlen=100, padding='post', truncating='post')

nn_model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=100),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_nn, y_train, epochs=5, validation_data=(X_test_nn, y_test))
# Αποθήκευση εκπαιδευμένου μοντέλου
joblib.dump(X_test_nn, 'X_test_nn.joblib')
joblib.dump(nn_model, 'nn_model.joblib')

# Εκτίμηση απόδοσης Logistic Regression
start_time = time.time()
y_pred_lr = logistic_regression_model.predict(X_test_tfidf)
end_time = time.time()
time_logistic_training = end_time - start_time
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print(f"Training Time: {time_logistic_training:.2f} seconds\n")

# Εκτίμηση απόδοσης SVM
start_time = time.time()
y_pred_svm = svm_model.predict(X_test_tfidf)
end_time = time.time()
time_svm_training = end_time - start_time
print("\nSVM Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))
print(f"Training Time: {time_svm_training:.2f} seconds\n")

# Εκτίμηση απόδοσης Νευρωνικού Δικτύου
start_time = time.time()
y_pred_nn = nn_model.predict(X_test_nn)
y_pred_nn = (y_pred_nn > 0.5).astype(int).flatten()
end_time = time.time()
time_nn_training = end_time - start_time

print("\nNeural Network Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Precision:", precision_score(y_test, y_pred_nn))
print("Recall:", recall_score(y_test, y_pred_nn))
print("F1 Score:", f1_score(y_test, y_pred_nn))
print(f"Training Time: {time_nn_training:.2f} seconds\n")

from math import comb

import numpy as np

# Συνάρτηση για αντικατάσταση των np.nan με ένα κείμενο
def replace_nan_with_text(text, replacement_text="[REPLACEMENT_TEXT]"):
    if pd.isna(text):
        return replacement_text
    return text


# Προετοιμασία 1000 τυχαίων κειμένων από το σύνολο ελέγχου
import random

random_texts = random.sample(df['text'].tolist(), 100)


start_time_inference = time.time()
i=0
# Κατηγοριοποίηση των 1000 τυχαίων κειμένων
for text in random_texts:
    # print(i)
    i = i+1
    # Μετατροπή του κειμένου σε διάνυσμα TF-IDF
    # Αντικατάσταση των np.nan στα κείμενα
    texts_without_nan = [replace_nan_with_text(text) for text in random_texts]
    text_tfidf = tfidf_vectorizer.transform(texts_without_nan)
    # Πρόβλεψη με το εκπαιδευμένο μοντέλο
    prediction = logistic_regression_model.predict(text_tfidf)

end_time_inference = time.time()

total_time = end_time_inference - start_time_inference
inference_time = (end_time_inference - start_time_inference) / len(random_texts)

print(f"Ο μέσος χρόνος κατηγοριοποίησης ενός κειμένου ήταν: {inference_time} δευτερόλεπτα")
print(f"Ο Συνολικος χρόνος κατηγοριοποίησης ενός κειμένου ήταν: {total_time} δευτερόλεπτα")

def output_lable(n):
    if n == 1:
        return "Fake News"
    elif n == 0:
        return "Not A Fake News"

# def manual_testing(news):
#     testing_news = {"text":[news]}
#     new_def_test = pd.DataFrame(testing_news)
#     new_def_test["text"] = new_def_test["text"].apply(wordopt)
#     new_x_test = new_def_test["text"]
#     print(new_x_test)
#     new_xv_test = tfidf_vectorizer.transform(new_x_test)
#     pred_LR = logistic_regression_model.predict(new_xv_test)
#     pred_SVN = svm_model.predict(new_xv_test)
#     # pred_NN = nn_model.predict(X_test_nn)
#     # pred_NN = (y_pred_nn > 0.5).astype(int).flatten()
#     # pred_NN = nn_model.predict(new_xv_test)

#     return print("\n\nLR Prediction: {} \nSVN Prediction: {}".format(output_lable(pred_LR[0]), output_lable(pred_SVN[0])))

# Φορτώνουμε το νέο κείμενο από ένα αρχείο (π.χ., 'news.txt')
# with open('fake_news.txt', 'r') as file:
#     news_text = file.read().replace('\n', '')

# manual_testing(news_text)