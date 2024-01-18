from flask import Flask, render_template, request
import joblib
import re
import pandas as pd
import string
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

logistic_regression_model = joblib.load('logistic_regression_model.joblib')
svm_model = joblib.load('svm_model.joblib')
nn_model = joblib.load('nn_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
tokenizer = joblib.load('tokenizer.joblib')

MAX_SEQUENCE_LENGTH = 100

def preprocess_text(text):
    text = wordopt(text)
    # Add zero-padding or truncation based on the length
    tokens = tokenizer.texts_to_sequences([text])
    padded_tokens = pad_sequences(tokens, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded_tokens

app = Flask(__name__)

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

def output_lable(n):
    if n == 1:
        return "Fake News"
    elif n == 0:
        return "Not A Fake News"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        testing_news = {"text":[news_text]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        # Load the TfidfVectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

        # Vectorize the input data
        new_x_test_tfidf = tfidf_vectorizer.transform(new_x_test)
        prediction_LR = logistic_regression_model.predict(new_x_test_tfidf)
        prediction_SVN = svm_model.predict(new_x_test_tfidf)
        print(prediction_LR)
        print(prediction_SVN)
        new_xv_test = preprocess_text(news_text)
        pred_NN = nn_model.predict(new_xv_test)
        pred_NN = (pred_NN > 0.5).astype(int).flatten()

        print(pred_NN)

        return render_template('index.html', prediction_LR=output_lable(prediction_LR), prediction_SVN=output_lable(prediction_SVN), pred_NN=output_lable(pred_NN), news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)