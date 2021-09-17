# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 19:27:45 2021

@author: xseber
"""

import pandas as pd
import joblib as jl
import urllib
import json
import numpy as np
import pythainlp
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request, render_template
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise
import joblib as jl
import tokenize
from utils import sp
source = pd.read_csv('static/data.csv')


def main():
    d = []
    for i in range(len(source)):
        a =  source['retrieval'].iloc[i]
        d.append(pythainlp.word_tokenize(a))
    
    tfigf = sk.feature_extraction.text.TfidfVectorizer(analyzer=sp)
    tokens_list_j = [','.join(tkn) for tkn in d]
    tfigf.fit(tokens_list_j)
    print(tfigf.transform(tokens_list_j))
    #corpus = tfigf.transform(tokens_list_j).toarray()
    #jl.dump(tfigf, 'static/vectorizer'+'.joblib')
    #jl.dump(tfigf.transform(tokens_list_j), 'static/corpus'+'.joblib')
    
if __name__ =='__main__':
    d = []
    for i in range(len(source)):
        a =  source['response'].iloc[i]
        d.append(pythainlp.word_tokenize(a))
    
    tfigf = sk.feature_extraction.text.TfidfVectorizer(analyzer=sp)
    tokens_list_j = [','.join(tkn) for tkn in d]
    tfigf.fit(tokens_list_j)
    print(tfigf.transform(tokens_list_j))
    #corpus = tfigf.transform(tokens_list_j).toarray()
    #jl.dump(tfigf, 'static/vectorizer'+'.joblib')
    #jl.dump(tfigf.transform(tokens_list_j), 'static/corpus'+'.joblib')