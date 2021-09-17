# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:10:48 2021

@author: xseber
"""

import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

source = pd.read_csv('static/data.csv')