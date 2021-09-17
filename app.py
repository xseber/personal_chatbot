# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:50:39 2021

@author: xseber
"""
import flask as f
import pandas as pd
import pythainlp
import numpy as np
import sklearn as sk

from sklearn.metrics import pairwise
import joblib as jl

from utils import sp
    
app = f.Flask(__name__, static_url_path='/static')
source = pd.read_csv('static/data.csv')
model = jl.load('static/vectorizer.joblib')
corpus = jl.load('static/corpus.joblib')
d = []
for i in range(len(source)):
    a =  source['retrieval'].iloc[i]
    d.append(pythainlp.word_tokenize(a))
    
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
    print(score)
    if score[0][np.argmax(score)] >0.5:
        response = source['response'].iloc[np.argmax(score)]
    else:
        response = 'ขอโทษนะ เรายังไม่รู้ว่าจะตอบอะไร'
    return response

if __name__ =='__main__':

    model = jl.load('static/vectorizer.joblib')
    corpus = jl.load('static/corpus.joblib')
    app.run(host='0.0.0.0', port=5000)

    