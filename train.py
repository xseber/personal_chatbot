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
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request, render_template
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise
import joblib as jl
import tokenize
source = pd.read_csv('static/data.csv')
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
    
if __name__ =='__main__':
    main()