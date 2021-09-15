# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:50:39 2021

@author: xseber
"""
import flask as f
import pandas as pd
import pythainlp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
import joblib as jl
def sp(x):
    return x.split(',')

def main():
    d = []
    for i in range(len(source)):
        a =  source['retrieval'].iloc[i]
        d.append(pythainlp.word_tokenize(a))
    
    tfigf = TfidfVectorizer(analyzer= sp)
    tokens_list_j = [','.join(tkn) for tkn in d]
    tfigf.fit(tokens_list_j)
    #corpus = tfigf.transform(tokens_list_j).toarray()
    jl.dump(tfigf, 'static/vectorizer'+'.joblib')
    
app = f.Flask(__name__, static_url_path='/static')
source = pd.read_csv('static/data.csv')

model = jl.load('static/vectorizer.joblib')
corpus = jl.load('static/corpus.joblib')

@app.route('/', methods=['GET','POST'])
def home():
    return f.render_template('home.html')

@app.route('/process_msg', methods=['GET','POST'])
def process():

    msg = f.request.form['msg']
    tokenize =[]
    tokenize.append(pythainlp.word_tokenize(msg))
    tokenize= [','.join(tkn) for tkn in tokenize]
    encode_msg = model.transform([msg]).toarray()
    score = pairwise.cosine_similarity(encode_msg, corpus)
    response = source['response'].iloc[np.argmax(score)]

    return response

if __name__ =='__main__':
    if False:main()
    app.run(port=8080)

    