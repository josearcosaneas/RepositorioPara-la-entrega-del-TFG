# -*- coding: 850 -*-
"""
Created on Sat May 30 20:58:04 2015

@author: blunt
"""
from summa import summarizer
from xml.dom.minidom import parse
import os
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

#from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score, KFold
############################################################################
############################################################################
ficherosT = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTest') # linux
ficherosE = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining')
materiasE = []
extractoE = []
materiasT = []
extractoT = []
############################################################################
############################################################################
def Replace(String):
    String = String.replace(" ","")
    return String         

############################################################################  
"""
Genera el tesauro
"""

def GeneraTarget(materias):
    for i in range(0,len(materias)):
        if len(materias[i])>0:
            materias[i] = materias[i].split(",")      
    materiasF=materias[0]
    for i in range(1,len(materias)):
        for j in range(0,len(materias[i])):
            if materias[i][j] not in materiasF:
                materiasF.append(materias[i][j])
    label=[]
    for i in range (1,len(materiasF)):    
        nuevo=str(materiasF[i])
        nuevo=Replace(nuevo)
        #print nuevo
        label.append(nuevo)
    label=list(set(label))
    return label   

# Lectura de ficheros del directorio
ficheros = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining') # linux
materias = []
# Extraccion de las materias por cada iniciativa del directorio
# las termino uniendo en una lista.
for i in ficheros:
    path="/home/blunt/Escritorio/iniciativas/iniciativasTraining/"+i
    midom=parse(path)    
    elements = midom.getElementsByTagName('materias')
    resultList = []
    if len(elements) != 0:
        for i in range(0,len(elements)):
            resultList.extend([elements[i].childNodes[0].nodeValue])
            materias.append(resultList[i])

target_names=GeneraTarget(materias)  
############################################################################
############################################################################
def leerTags(path,tag):
    midom=parse(path)
    elements = midom.getElementsByTagName(tag)
    resultList1 = []

    if len(elements) > 0:
        for i in range(0,len(elements)):
            resultList1.extend([elements[i].childNodes[0].nodeValue])
    return resultList1
"""
summarizer.summarize(texto,lenguaje,ratio,words)
"""    
def resumir(texto,lenguaje='spanish',ratio=0.2):
    if not(lenguaje):
        return summarizer.summarize(texto, language='spanish',ratio=ratio)
    else:
        return summarizer.summarize(texto, language=lenguaje,ratio=ratio)
############################################################################
############################################################################   
        
 
"""
Funcion encargada de transformar materias para su entrada 
"""
def transformaMaterias(materiasF):
    label=[]
    materiasF=materiasF[0].split(",")

    for i in range (1,len(materiasF)):  

        nuevo=str(materiasF[i])
        nuevo=Replace(nuevo)
        label.append(nuevo)

    return label  
"""
Preparacion de materias para pasar al clasificador 
"""    
def PreparaMaterias(materias):
    materiasTrain = []
    for i in materias:
        if len(i)>0:
            i=transformaMaterias(i)
        materiasTrain.append(i)
    
    return materiasTrain
############################################################################
def tokenize(resultList1):
    entrada=[]
    tokens = word_tokenize(resultList1)
    filtered_words = [w for w in tokens if not w in stopwords.words('spanish')]

    stemmer = SnowballStemmer('spanish')
    for i in filtered_words:
        stri = unicode(i,errors='replace')
        entrada.append(stemmer.stem(stri))

    return entrada
############################################################################
# Leemos los de entrenamiento
for i in ficherosE:
    path="/home/blunt/Escritorio/iniciativas/iniciativasTraining/"+i
    print path
    materiasE.append(leerTags(path,'materias'))
    parrafos = leerTags(path,'parrafo')
    todo=""
    for i in range(0,len(parrafos)):
        todo = todo +" "+str(parrafos[i])    
    try:
#        resumir(todo)
        todo = resumir(todo,ratio=0.25)
        todo= tokenize(todo) 
    
        final = ""    
        for i in range(0, len (todo)):
            final = final+" "+str(todo[i])    
        final =str(final)
        extractoE.append(final)
    except ValueError:
        print "error el archivo: "+ path +"No se puede resumir probablemante porque este vacio. "
                                        
        #os.remove(path)
    #todo = resumir(todo)
    
############################################################################
for i in ficherosT:
    path="/home/blunt/Escritorio/iniciativas/iniciativasTest/"+i
    materiasT.append(leerTags(path,'materias'))
    parrafos = leerTags(path,'parrafo')
    todo=""
    for i in range(0,len(parrafos)):
        todo = todo +" "+str(parrafos[i])
        
    try:

        todo = resumir(todo,ratio=0.25)
        todo= tokenize(todo) 
    
        final = ""    
        for i in range(0, len (todo)):
            final = final+" "+str(todo[i])    
        final =str(final)
        extractoT.append(final)
    except ValueError:
        print "error el archivo: "+ path +" no se puede resumir probablemante porque este vacio. "
############################################################################
y_test = PreparaMaterias(materiasT)
y_train = PreparaMaterias(materiasE)
iniciativasTraining = np.array(extractoE)
iniciativasTest= np.array(extractoT)
############################################################################
############################################################################
def clasificador(X_train, y_train, X_test, target_names):
    
    lb = preprocessing.MultiLabelBinarizer()
    
    Y = lb.fit_transform(y_train)
    
    classifier = Pipeline([
        ('vectorizer',CountVectorizer(strip_accents='unicode')),
        ('tfidf',TfidfTransformer()),
        ('clf',OneVsRestClassifier(RandomForestClassifier(n_estimators=20,random_state=42)))])
       
    f = open("resultadosResumenRF-0.25.txt","w")
    f.write(" Resultados Precision/Recall/F1\n\n ")
    cv = KFold(Y.shape[0], n_folds=10, shuffle=True, random_state=42)
################################################################################################################       
    """
    Resultados de las 10 validacion metrica Precision
    """   
################################################################################################################
    scores = cross_val_score(classifier, X_train, y_train, cv=cv)
#        print("Precision\n.")
    print(scores)
    #print("Precision_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#    f.write("Precision\n")
    f.write(str(scores)+"\n")
         
    classifier.fit(X_train,Y)
    predicted = classifier.predict(X_test)
    all_label = lb.inverse_transform(predicted)
    print all_label        
############################################################################    
clasificador(iniciativasTraining, y_train, iniciativasTest, target_names)
