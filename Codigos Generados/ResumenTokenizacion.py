#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Created on Fri Aug 21 11:29:43 2015

@author: blunt
"""


from xml.dom.minidom import parse
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from xml.dom.minidom import parse
import numpy as np
from nltk.corpus import stopwords
from summa import summarizer

texto='Tengo un equipo de Informáticos, formado por informáticos e informáticas. Correo: TFG@correo.ugr.es'

tokens = word_tokenize(texto)
print "tokens"
print tokens

filtered_words = [w for w in tokens if not w in stopwords.words('spanish')]
filtered2 =[]
sen = ['El','La','Los','Las',':',',','.','desde' , 'para', 'por', 'a' , 'ante','bajo','con','contra','en','un',"segun","sin",'sobre','tras']
for i in range(0,len(filtered_words)):
    if filtered_words[i] not in sen:
        filtered2.append(filtered_words[i])

print "stop_words"
print filtered2
stemmer = SnowballStemmer('spanish')

stems=[]
j=0
for i in range(0,len(filtered2)):
        
    stems.append( str(stemmer.stem(filtered2[i])))
print "stems"
print stems