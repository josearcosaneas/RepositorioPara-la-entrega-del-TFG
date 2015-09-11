# -*- coding: 850 -*-
"""
Created on Sat May 30 23:59:40 2015

@author: blunt
"""

from summa import summarizer
from xml.dom.minidom import parse
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support

############################################################################
############################################################################

direccionTest = '/home/blunt/Escritorio/iniciativas/iniciativasTest'
direccionEntrenamiento = '/home/blunt/Escritorio/iniciativas/iniciativasTraining'

class Clasificador():

    def __main__(self,direccionTest,direccionEntrenamiento):
            
#        ficherosT = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTest') # linux
#        ficherosE = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining')
        ficherosT = os.listdir(direccionTest) # linux
        ficherosE = os.listdir(direccionEntrenamiento)
        materiasE = []
        extractoE = []
        materiasT = []
        extractoT = []
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
    ## n_jobs para paralelizar 
    ## min_samples_split minimo de ejemplo con lo que se terminar  el arbol
    ##max_depth maxima profundidad en el caso en que sea nula el arbol descendera 
    ## hasta el valor indicado por min_samples_split
    ############################################################################   
    def clasificador(self,X_train, y_train, X_test, target_names, y_test,all_labels):
        
        lb = preprocessing.MultiLabelBinarizer()
        lb2=preprocessing.MultiLabelBinarizer()
        Y = lb.fit_transform(y_train)
        y_test = lb2.fit_transform(y_test)
        
        print y_test
        classifier = Pipeline([
            ('vectorizer',CountVectorizer(strip_accents='unicode')),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(RandomForestClassifier(n_estimators=15,random_state=42)))])
            
#        f = open("resultadosExtractoRF.txt","w")
#        f.write(" Resultados Precision/Recall/F1\n\n ")
#        cv = KFold(Y.shape[0], n_folds=10, shuffle=True, random_state=42)
#################################################################################################################       
#        """
#        Resultados de las 10 validacion metrica Precision
#        """   
#################################################################################################################
#        scores = cross_val_score(classifier, X_train, y_train, scoring='precision_micro', cv=cv)
##        print("Precision\n.")
##        print(scores)
#        #print("Precision_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        f.write("Precision\n")
#        f.write(str(scores)+"\n")
#        f.write(str("Precision_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
#        
#        scores = cross_val_score(classifier, X_train, y_train, scoring='precision_macro', cv=cv)
##        print("CV scores.")
##        print(scores)
##        print("Precision_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        
#        f.write(str(scores)+"\n")
#        f.write(str("Precision_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))    
#################################################################################################################        
#        """
#        Resultados de las 10 validacion metrica Recall
#        """        
#################################################################################################################        
#        f.write("Recall\n")
#        scores = cross_val_score(classifier, X_train, y_train, scoring='recall_micro', cv=cv)
##        print("CV scores.")
##        print(scores)
##        print("recall_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        f.write(str(scores)+"\n")
#        f.write(str("Recall_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
#        
#        scores = cross_val_score(classifier, X_train, y_train, scoring='recall_macro', cv=cv)
##        print("CV scores.")
##        print(scores)
##        print("Recall_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        f.write(str(scores)+"\n")
#        f.write(str("Recall_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
#################################################################################################################        
#        """
#        Resultados de las 10 validaciones para F1
#        """
#################################################################################################################       
#        f.write("F1\n")
#        scores = cross_val_score(classifier, X_train, y_train, scoring='f1_micro', cv=cv)
##        print("CV scores.")
##        print(scores)
##        print("F1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        f.write(str(scores)+"\n")
#        f.write(str("F1_micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
#        scores = cross_val_score(classifier, X_train, y_train, scoring='f1_macro', cv=cv)
##        print("CV scores.")
##        print(scores)
##        print("F1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        f.write("\n"+str(scores)+"\n")
#        f.write(str("F1_macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
#################################################################################################################
#        f.close()        
#        scores = cross_val_score(classifier, X_train, y_train, scoring='f1_weighted', cv=cv)
#        print("CV scores.")
#        print(scores)
#        print("F1_weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        
#        scores = cross_val_score(classifier, X_train, y_train, scoring='recall_weighted', cv=cv)
#        print("CV scores.")
#        print(scores)
#        print("Recall_weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#        scores = cross_val_score(classifier, X_train, y_train, scoring='precision_weighted', cv=cv)
#        print("CV scores.")
#        print(scores)
#        print("Precision_weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#     
        classifier.fit(X_train,Y)
        predicted = classifier.predict(X_test)
       # print predicted
        #y_test = lb2.fit_transform(y_test)
        
       # precision, recall, f, _ = precision_recall_fscore_support(y_test, predicted,average = "micro")
       
        etiquetas = lb.inverse_transform(predicted)

        print etiquetas        
    ############################################################################            
     
    """
    Funcion encargada de transformar materias para su entrada 
    """
    def transformaMaterias(self,materiasF):
        label=[]
        materiasF=materiasF[0].split(",")
        
        for i in range (1,len(materiasF)):  
    
            nuevo=str(materiasF[i])
            #nuevo=self.Replace(nuevo)
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
    
    
clasificador= Clasificador()
clasificador.__main__(direccionTest,direccionEntrenamiento)

