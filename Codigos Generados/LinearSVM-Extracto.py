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
#from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.cross_validation import cross_val_score, KFold
#
#
#from sklearn.metrics import precision_recall_fscore_support

############################################################################
############################################################################

direccionTest = '/home/blunt/Escritorio/iniciativas/iniciativasTest'
direccionEntrenamiento = '/home/blunt/Escritorio/iniciativas/iniciativasTraining2'

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
        ficheros = os.listdir('/home/blunt/Escritorio/iniciativas/iniciativasTraining2') # linux
        materias = []
        # Extraccion de las materias por cada iniciativa del directorio
        # las termino uniendo en una lista.
        for i in ficheros:
            path="/home/blunt/Escritorio/iniciativas/iniciativasTraining2/"+i
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
    ############################################################################   
    def clasificador(self,X_train, y_train, X_test, target_names, y_test,all_labels):
        
        lb = preprocessing.MultiLabelBinarizer()
        #lb2=preprocessing.MultiLabelBinarizer()
#        y_train = lb.fit_transform(y_train)
        y_train = lb.fit_transform(y_train)
        
        classifier = Pipeline([
            ('vectorizer',CountVectorizer(strip_accents='unicode')),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(SVC(kernel='linear',C=1.0, cache_size=200, coef0=0.0, degree=3,
                                               gamma=0.0, max_iter=-1,  random_state=None,
                                               shrinking=True, tol=0.001, verbose=False,probability=True)))])


        
        classifier2 = Pipeline([
            ('vectorizer',CountVectorizer(strip_accents='unicode')),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.001, C=1.0, multi_class='ovr',
                                                 fit_intercept=True, intercept_scaling=1, 
                                                 class_weight=None, verbose=0, random_state=None, max_iter=2000)))])
            

        classifier.fit(X_train,y_train)
        probabilidad = classifier.predict_proba(X_test)
       
        classifier2.fit(X_train,y_train)


        classifier2.fit(X_train,y_train)
#        print classifier2.decision_function(X_test)
        predicted= lb.inverse_transform(classifier2.predict(X_test))


        
#        print predicted

###############################################################
## Probabilidades respecto de las clases. 
##      
#        f = open("Probabilidades","w")
#        f.write("PROBABILIDADES\n")

        for i in range(0,len(probabilidad)):
            print " *"*30
            print  " * "*30
            print " Etiquetas reales"
            print str(y_test[i])
            print " * "*30
            
            print "Etiquetas predecidas"
            
            print str(predicted[i])
            print " * "*30
            print 
            for j in range (0, len(list(lb.classes_))):
      
                label= lb.classes_[j]
                nlabel= classifier2.classes_[j]
                prob=probabilidad[i][j]
                if prob > 0.02:                  
                      print str(label) + " -> " + str(nlabel) +" -> " +str(prob )
       

        
        
#        
#        for i in range(0,len(probabilidad)):
#            f.write ("Etiquetas Predecidas\n")
#            f.write("***"*20)
#            f.write("\n")
#            f.write(str(predicted[i]))
#            f.write("\n")
#            f.write("***************")
#            f.write(str(y_test[i]))
#            f.write("\n")
#            f.write("****"*20)
#            f.write("\n")
#            f.write( "Etiquetas juntos a su valor numerico y su probabilidad")
#            f.write("\n")
#            for j in range (0, len(list(lb.classes_))):
#                f.write( "*"*20)        
#                label= lb.classes_[j]
#                nlabel= classifier2.classes_[j]
#                prob=probabilidad[i][j]
#                f.write("\n")
#                f.write(str(label))
#                f.write(" -> ")                
#                f.write(str(nlabel))                
#                f.write(" -> ")    
#                f.write(str(prob))
#       
#                f.write("\n")
###############################################################
#        for i in range(0,len(list(lb.classes_))):
#            print lb.classes_[i], classifier2.classes_[i], probabilidad[0][i]


            
###############################################################
       # precision, recall, f, _ = precision_recall_fscore_support(y_test, predicted,average = "micro")
#        f = open("resultadosExtractoProbabilidades.txt","w")
#        f.write("Probabilidades \n\n")
#        etiquetas = lb.inverse_transform(predicted)
#        for i in range(0,len(probabilidad)):
#                f.write( str(probabilidad[i]))
#                f.write("\n")
                
#        f = open("resultadosExtracto.txt","w")
#        f.write(" Resultados\n\n ")
#        f.write(str(probabilidad ))
#        
#        f.close()
#                
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

