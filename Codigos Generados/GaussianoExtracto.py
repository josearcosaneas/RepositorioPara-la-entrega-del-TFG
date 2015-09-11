

# -*- coding: 850 -*-
"""
Created on Sat May 30 23:59:40 2015

@author: blunt
"""
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import random
from summa import summarizer
from xml.dom.minidom import parse
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
#from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.cross_validation import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from random import sample

class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

direccionTest = '/home/blunt/Escritorio/iniciativas/iniciativasTest'
direccionEntrenamiento = '/home/blunt/Escritorio/iniciativas/iniciativasTraining'
def resumir(texto,lenguaje='spanish',ratio=0.2):
    if not(lenguaje):
        return summarizer.summarize(texto, language='spanish',ratio=ratio)
    else:
        return summarizer.summarize(texto, language=lenguaje,ratio=ratio)
def tokenize(resultList1):
    entrada=[]
    tokens = word_tokenize(resultList1)
    filtered_words = [w for w in tokens if not w in stopwords.words('spanish')]

    stemmer = SnowballStemmer('spanish')
    for i in filtered_words:
        stri = unicode(i,errors='replace')
        entrada.append(stemmer.stem(stri))

    return entrada
        

class Clasificador():

    def __main__(self,FicherosT,FicherosE):
            
        ficherosT = FicherosT # linux
        ficherosE = FicherosE
        materiasE = []
        extractoE = []
        materiasT = []
        extractoT = []
#        resumenes = []
        # Leemos los de entrenamiento
        for i in ficherosE:
            path=direccionEntrenamiento+"/"+i
            materiasE.append(self.leerTags(path,'materias'))
            
            extracto = self.leerTags(path,'extracto')
            todo = str(extracto)
            extractoE.append(todo)
        ############################################################################
        for i in ficherosT:
            path=direccionTest+"/"+i
            materiasT.append(self.leerTags(path,'materias'))
           
            extracto = self.leerTags(path,'extracto')
            todo = str(extracto)
            extractoT.append(todo)
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
                final =str(final)##remune
                resumenes.append(final)
            except ValueError:
                print "error el archivo: "+ path +"No se puede resumir probablemante porque este vacio. "                
        ############################################################################
        #y_test = PreparaMaterias(materiasT)
   
        y_test = self.PreparaMaterias(materiasT)
        y_train = self.PreparaMaterias(materiasE)
        iniciativasTraining = np.array(extractoE)
        iniciativasTest= np.array(extractoT)
        ############################################################################
        ############################################################################
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
        
        target_names=self.GeneraTarget(materias)          
        print "Clasifcandod"
        self.clasificador(iniciativasTraining, y_train, iniciativasTest, target_names, y_test,target_names)        
  
    def Replace(self,String):
        String = String.replace(" ","")
        return String         
    
    ############################################################################  
    """
    Genera el tesauro
    """    
    def GeneraTarget(self,materias):
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
            nuevo=self.Replace(nuevo)
            #print nuevo
            label.append(nuevo)
        label=list(set(label))
        return label     
    ############################################################################
    ############################################################################
    
    def leerTags(self,path,tag):
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
    def resumir(self,texto,lenguaje='spanish',ratio=0.5):
        if not(lenguaje):
            return summarizer.summarize(texto, language='spanish',ratio=ratio)
        else:
            return summarizer.summarize(texto, language=lenguaje)

    ############################################################################
    ############################################################################   
    def clasificador(self,X_train, y_train, X_test, target_names, y_test,all_labels):
        
        lb = preprocessing.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train)
        
        classifier = Pipeline([
            ('vectorizer',CountVectorizer(strip_accents='unicode')),
            ('tfidf',TfidfTransformer()),
            ('to_dense', DenseTransformer()),
            ('clf',OneVsRestClassifier(GaussianNB()))])
            

     
        classifier.fit(X_train,Y)
        
        predicted = classifier.predict(X_test)


        etiquetas = lb.inverse_transform(predicted)

                
        for i in range(0,len(etiquetas)):
            etiquetas[i]=list(etiquetas[i])

        
        valoresMacro = self.macro(etiquetas,y_test)
        valoresMicro = self.micro(etiquetas, y_test)        
        
        
        
    def macro(self,predecido,y_test):
        PrecisionTotal = 0
        RecallTotal = 0
        F1Total = 0
        
        for i in range(0, len(predecido)):
            
            TP =len(list(set(predecido[i]) & set(y_test[i])))
            TP=float(TP)
            #print TP
            FP =len(list(set(predecido[i]) - set(y_test[i])))
            FP = float(FP)        
            #print FP
            FN =len(list(set(y_test[i]) -  set(predecido[i])))
            #print FN
            FN=float(FN)
            #print FP+FN
            if FP+TP != 0.0:
                precision = float(TP)/(TP+FP)
            else:
                precision = 0.0
            if FP+FN != 0.0 and TP != 0:
                recall = float((float(TP)/TP+FN))
            else:
                recall=0.1
            F1 = 2*((precision*recall)/(precision+recall))
            
            
            PrecisionTotal = PrecisionTotal + precision
            RecallTotal = RecallTotal + recall
            F1Total = F1Total+F1 

        PrecisionTotal = PrecisionTotal /float(len(predecido))
        RecallTotal =RecallTotal /float(len(predecido))
        F1Total = F1Total /float(len(predecido))
        
        print "PrecisionMacro " + str(PrecisionTotal)
        print "RecallMAcro " + str (RecallTotal)            
        print "F1Macro " +str(F1Total)
        
        return PrecisionTotal, RecallTotal, F1Total
        
    def micro(self,predecido, y_test):
        
        TPt=0
        FPt=0
        FNt=0
        
    
        for i in range(0, len(predecido)):
            
            TP =len(list(set(predecido[i]) & set(y_test[i])))
            TP=float(TP)
            #print TP
            FP =len(list(set(predecido[i]) - set(y_test[i])))
            FP = float(FP)        
            #print FP
            FN =len(list(set(y_test[i]) -  set(predecido[i])))
            #print FN
            FN=float(FN)
            
            TPt = TPt+TP
            FPt = FPt + FP
            FNt = FNt + FN
            
        if (TPt + FPt) !=0.0 :
            precision = float(TPt)/ (TPt + FPt)
        else:
            precision = 0.0
        if FPt+FNt != 0.0:
            recall = (float(TPt/(TPt+FNt)))
        else:
            recall=0.1
    
        if precision+recall != 0.0:
            F1 = 2*((precision*recall)/(precision+recall))
        else :
            F1 = 0
        print "Precision Micro " + str(precision)
        print "Recall micro "+  str(recall) 
        print  "F1 micro " +str(F1)    
        
        return precision,recall,F1        
        
        
        
        
        
    ############################################################################            
     
    """
    Funcion encargada de transformar materias para su entrada 
    """
    def transformaMaterias(self,materiasF):
        label=[]
        materiasF=materiasF[0].split(",")
        
        for i in range (1,len(materiasF)):  
    
            nuevo=str(materiasF[i])
            
            label.append(nuevo)
    
        return label  
    """
    Preparacion de materias para pasar al clasificador 
    """    
    def PreparaMaterias(self,materias):
        materiasTrain = []
        for i in materias:
            if len(i)>0:
                i=self.transformaMaterias(i)
            materiasTrain.append(i)
        
        return materiasTrain
    


def cross_validation(ficheros,kfols):
    FicherosT =[]
    FicherosE = []

    N=len(ficheros)
    
    
    elementos = N/kfols
      
    for i in range(0,kfols):
        test = []   
        entrenamiento = ficheros
        semilla= (i*89)+353
        random.seed(semilla)
        indicesEntrenamiento = sample(range(0, N-1), elementos)
        for i in range(0,len(indicesEntrenamiento)):
            test.append(ficheros[indicesEntrenamiento[i]])
        entrenamiento= list(set(entrenamiento) - set(test))
        FicherosT.append(test)
        FicherosE.append(entrenamiento)
     
    return FicherosT,FicherosE   
     
ficheros = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining')
kfols=15
FicherosT,FicherosE=cross_validation(ficheros,kfols)    

clasificador= Clasificador()
print " Clasificando"
for i in range (0,len(FicherosT)):
    clasificador.__main__(FicherosT[i],FicherosE[i])

