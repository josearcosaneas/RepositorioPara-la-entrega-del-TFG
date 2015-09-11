#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: José Arcos Aneas

Archivo de codigo donde de desarrollará la aplicación.

"""
import wx
import pickle
from Tkinter import *
from summa import summarizer
from xml.dom.minidom import parse
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from subprocess import call
from nltk.corpus import stopwords
from nltk import word_tokenize

APP_TITLE = "Ayuda Clasificador Automático"
MENU_ABOUT = "&Acerca de"
STATUS_ABOUT = "Información sobre el programa"
MENU_SALIR = "&Salir"
STATUS_SALIR = "Finaliza el programa"
MENU_TITLE = "&Documento"
ABOUT_TITLE = "Acerca de"
ABOUT_CONTENT = """Clasificador de iniciativas multilabel. Trabajo Fin de Carrera UGR"""
TEMPLATE = """
 INTRODUCCION

    La aplicación consta de cuatro ventanas principales que permiten llevar a 
    cabo la clasificación de un conjunto de iniciativas.

    La aplicación se inicia con una ventana principal que da acceso a dos ventanas
    distintas.

    Por un lado, es posible generar un modelo a partir de un conjunto de entrenamiento
    y guardarlo para su posterior uso.

    Una vez tengamos entrenado algún modelo; será posible cargarlo y hacer uso de él 
    para clasificar un conjunto de iniciativas.


 VENTANA PRINCIPAL


  Esta ventana es la ventana principal del programa, desde la que podremos 
  elegir cuatro opciones:

   1. Entrenar clasificador 
        Esta opción permite la creación de un clasificador, guardar el modelo para poder
        ser usado desde la vetana cargar modelo.
   2. Cargar modelo
        Esta opción permite cargar un modelo de un clasificador previamente entrenado.
        También permite la clasificación de iniciativas.
   3. Ayuda 
        Este botón muestra la pantalla actual.
   4. Cerrar.
        Cierra la aplicación.

 ENTRENAR CLASIFICADOR
   
   Esta ventana se inicia al clicar en el botón "Entrenar Clasificador".
   La ventana ofrece una serie de opciones que dan la posibilidad cargar un directorio
   de entrenamiento, generar un clasificador a partir de este y guardar el modelo, 
   
   Los botones y su funcionalidad se muestran a continuación:
   
   1. Directorio Entrenamiento.
        Con este botón elegimos el directorio que contendrá las iniciativas que 
        servirán para entrenar un clasificador.
        Aunque no es necesario, se aconseja sea lo que se realice en primer lugar.
        Tenga en cuenta que el tiempo que se puede dedicar a esta tarea dependerá
        del número de iniciativas que se empleen en el entrenamiento.
                
   2. Guardar modelo.
        Con este boton podemos indicar donde y con que nombre guardar el modelo
        generado, para poder rehutilizarlo sin necesidad de entrenar un clasificador.
                           
   3. Entrenar
        Al clicar en este boton, indicamos que deseamos generar un modelo
        de nuestro clasificador a partir del conjunto de entrenamiento
        que debemos elegir previamente.
        Una vez se haya realizado la clasificación, emergerá de manera sucesiva
        una ventana por cada una de las iniciativas que se desean clasificar,
        donde será posible la elección de las etiquetas con las que clasificar
        una iniciativa.

    5. Cerrar 
        Este botón cierra la ventana actual y nos devuelve a la ventana inicial.
 
 CARGAR MODELO
    
    Para acceder a esta ventana será necesario clicar en el botón "Cargar
    modelo" desde la ventana inicial.
    
    Una vez desplegada la ventana serán posibles una serie de opciones para 
    realizar la clasificación de un conjunto de iniciativas.
    
    A diferencia de la ventana anterior será posible cargar un modelo y reducir 
    el tiempo dedicado a la clasificación.
    
    Los distintos botones disponibles en esta ventana junto a su funcionalidad 
    se muestran a continuación.
    A continuación se describe la funcionalidad de cada uno de los diferentes 
    botones disponibles en esta ventana.
    
   1. Elegir modelo.
        Este botón permite cargar un archivo que contenga un modelo de un 
        clasificador previamente entrenado.
        Esta opción reduce el coste de tiempo en llevar a cabo la clasificación.
        
   2. Directorio clasificación
        Este botón permite cargar un directorio que contendrá las iniciativas
        que se desean clasificar. 

   3. Clasificar 
        En el caso en que se haya elegido un modelo y un directorio de 
        clasificación, esta opción permite clasificar el conjunto de iniciativas.
        Una vez se haya realizado la clasificación, emergerá de manera sucesiva
        una ventana por cada una de las iniciativas que se desean clasificar,
        donde será posible la elección de las etiquetas con las que clasificar
        una iniciativa.

   4. Cerrar.
        Botón que permite cerrar la ventana actual.
         
 VENTANA ELECCIONES
    
   Esta ventana permite la elección de una serie de etiquetas, previamente 
   seleccionadas por el clasificador, distinguiendo entre las que poseen mayores 
   posibilidades de ser apropiadas y las que poseen una posibilidad menor.
   Esta ventana mostrará el contenido de parte de la iniciativa, además de una 
   serie de opciones para hacer más fácil la clasificación.
   
   
   1. Guardar
        Esta opción salva las materias seleccionadas dentro del campo materias
        de la iniciativa a la que corresponda. En caso de no contener el nodo 
        materias lo creará. En caso de existir el campo materias incluirá el
        contenido seleccionado junto al ya existente.
        
   2. Visualizar iniciativa.
        Este botón permite visualizar una iniciativa. El entorno para la 
        visualización será firefox.

   3. Siguiente.
        Este botón estará visible siempre que no estemos en la última iniciativa
        de las que nos disponemos a clasificar. 
        Permite pasar a la página de selección de la próxima iniciativa.
   
   4. Cerrar.
        Será visible en la última iniciativa y permite cerrar la página actual.
        Nos dirige a la página "Cargar modelo".
    
"""
"""
Ventana de ayuda para el usuario

"""
class VentanaAyuda(wx.Frame):

    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(600, 600))
        self.contenido = wx.TextCtrl(self, style=wx.TE_MULTILINE,
                                     value=TEMPLATE)
        self.CreateStatusBar()  # Crea una barra de estado
        # Inicializa un menú
        filemenu = wx.Menu()
        # Crea items del menú
        menu_about = filemenu.Append(wx.ID_ABOUT, MENU_ABOUT, STATUS_ABOUT)
        menu_exit = filemenu.Append(wx.ID_EXIT, MENU_SALIR, STATUS_SALIR)
        # Crea la barra de menú
        menubar = wx.MenuBar()
        menubar.Append(filemenu, MENU_TITLE)  # Titulo del menu
        self.SetMenuBar(menubar)  # Agrega la barra de menu al frame
        # Establece eventos
        self.Bind(wx.EVT_MENU, self.on_about, menu_about)
        self.Bind(wx.EVT_MENU, self.on_exit, menu_exit)
        self.Bind(wx.EVT_CLOSE, self.on_exit)
        self.Centre(True)  # Centrar la ventana en pantalla
        self.Show(True)  # Mostrar la ventana


    def on_about(self, event):
        """Mostrar un diálogo acerca de"""
        dialog = wx.MessageDialog(self, ABOUT_CONTENT, ABOUT_TITLE, wx.OK)
        dialog.ShowModal()  # mostrar diálogo
        dialog.Destroy()  # finalizar diálogo

    def on_exit(self, evt):
        """Salir del programa"""
        self.MakeModal(False)
        self.Destroy()  # Cierra la ventana

"""
Ventana desarrollada para permitir crear un modelo para nuestro clasificador.
"""
class VentanaEntrenar(wx.Frame):
    
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, wx.NewId(),"Ventana de entrenamiento", size=(700,350))
        self.Bind(wx.EVT_CLOSE, self.cerrar)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour("white")
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.img = wx.EmptyImage(100,100)
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, 
                                         wx.BitmapFromImage(self.img))
        self.imageCtrl.SetPosition((150,100))
        self.img = wx.Image("images.jpg", wx.BITMAP_TYPE_ANY)
        self.imageCtrl.SetBitmap(wx.BitmapFromImage(self.img))
        # Botones
        self.botonElegirDirectorioEntrenamiento = wx.Button(self, wx.NewId(),"Elegir directorio Entrenamiento",pos=(3,300))


        self.botonSalir = wx.Button(self,wx.NewId(),"Cerrar",pos=(600,300))
        self.botonEntrenar = wx.Button(self,wx.NewId(),"Entrenar",pos=(450,300))
        
        self.botonGuardarModelo = wx.Button(self,wx.NewId(),"Guardar Modelo",pos=(303,300))
        # Posicionanmiento De botones
   
        self.botonGuardarModelo.Bind(wx.EVT_BUTTON,self.botonClickGuardarModelo)
        self.botonElegirDirectorioEntrenamiento.Bind(wx.EVT_BUTTON, self.ClickElegirDirectorioEntrenamiento)
        self.botonSalir.Bind(wx.EVT_BUTTON, self.botonClickSalir)
        self.botonEntrenar.Bind(wx.EVT_BUTTON,self.ClickBotonEntrenar)

        self.Layout()
      

    def ClickBotonEntrenar(self, evt):

        self.clasificador = Clasificador()
        self.modelo= self.clasificador.entrenar(_selectedDirEntrenamiento)
        print _selectedDirEntrenamiento
        pathModelo =""            
        for i in range(1, len(_selectedDirEntrenamiento.split("/"))-1):
             pathModelo = pathModelo +"/"+_selectedDirEntrenamiento.split("/")[i]
           
        pathModelo = pathModelo +"/modeloClases.txt"
        print pathModelo
        f = open(pathModelo,"w")
        f.write(_selectedDirEntrenamiento)
        f.close()
        
    def ClickElegirDirectorioEntrenamiento(self, evt):
        
        global _selectedDirEntrenamiento 
        
        userPath = ''
        app = wx.App()
        dialog = wx.DirDialog(None, "Por favor eliga un directorio para entrenamiento:",\
            style=1 ,defaultPath=userPath, pos = (10,10))
        if dialog.ShowModal() == wx.ID_OK:
            _selectedDirEntrenamiento = dialog.GetPath()
            print _selectedDirEntrenamiento
            pathModelo =""            
            for i in range(1, len(_selectedDirEntrenamiento.split("/"))-1):
                pathModelo = pathModelo +"/"+_seletedDirEntrenamiento.split("/")[i]
            
            pathModelo = pathModelo +"/modeloClases.txt"
            print pathModelo
            f = open(pathModelo,"w")
            f.write(_selectedDirEntrenamiento)
            f.close()
            return _selectedDirEntrenamiento
        else:
            app.skip()
    
    def cerrar(self, evt):
          
        self.MakeModal(False)
        evt.Skip()  
        
    def botonClickSalir(self, evt):
        self.MakeModal(False)
        self.Close()
    
    def botonClickGuardarModelo(self,evt):
        ventana = wx.FileDialog(None, message="Salvar como",
                                defaultDir=os.getcwd(),defaultFile="",wildcard = '*.*',
        style=wx.SAVE| wx.CHANGE_DIR)
        if ventana.ShowModal() == wx.ID_OK:
            self.fichero = ventana.GetPath()
            f = open(self.fichero,"w")
            f.write(self.modelo)
            f.close()
            print "EL ARCHIVO A SIDO GUARDADO CON EXITO"
        ############################################
        else: self.fichero = None
        # Destruimos el diálogo.
        ventana.Close()

"""
La siguiente clase implementa la ventana principal.
Permite dirigirnos la ventana de entrenamiento y a la ventana de clasificación
"""
    
class VentanaPrincipal(wx.Frame):
    def __init__(self):
        
        wx.Frame.__init__(self, None, wx.NewId(), "Clasificador de iniciativas parlamentarias",
                          pos=(100,100), size=(500,350))
        self.SetBackgroundColour("white")
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.img = wx.EmptyImage(100,100)
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, 
                                         wx.BitmapFromImage(self.img))
        self.imageCtrl.SetPosition((40,120))
        self.img = wx.Image("images.jpg", wx.BITMAP_TYPE_ANY)

 
        self.imageCtrl.SetBitmap(wx.BitmapFromImage(self.img))
        
        
        
        self.botonEntrenarClasificador = wx.Button(self,wx.NewId(),"Entrenar Clasificador",pos=(3,300))

        self.botonCargarClasificador =wx.Button(self,wx.NewId(),"Cargar Clasificador",pos=(163,300))
        self.botonSalir = wx.Button(self,wx.NewId(),"Cerrar",pos=(400,300))
        self.botonInformacion = wx.Button(self, wx.NewId(),"Ayuda",pos=(400,10))

        self.botonCargarClasificador.Bind(wx.EVT_BUTTON, self.botonClickCargarClasificador)
        self.botonEntrenarClasificador.Bind(wx.EVT_BUTTON, self.botonClickEntrenarClasificador)        
        self.botonInformacion.Bind(wx.EVT_BUTTON, self.botonClickInfo)
        self.botonSalir.Bind(wx.EVT_BUTTON, self.botonClickSalir)

        self.Layout()

    def botonClickCargarClasificador(self, evt):
        ventana = VentanaEntrenado(self)
        ventana.Show(True)
        ventana.MakeModal(True)
        
    def botonClickEntrenarClasificador(self,evt):
        ventana = VentanaEntrenar(self)
        ventana.Show(True)
        ventana.MakeModal(True)

    def botonClickInfo(self, evt):
        ventana = VentanaAyuda(self,APP_TITLE)
        ventana.Show(True)
        ventana.MakeModal(True)
 
    def botonClickSalir(self, evt):
           
        self.MakeModal(False)
        self.Destroy()  
      
"""
A continuación se implementan algunas funciones que pueden sernos de interes
Estas funciones son similares a las que pertenecen a algunas de las clases.

"""
def leerTags(path,tag):
    midom=parse(path)
    elements = midom.getElementsByTagName(tag)
    resultList1 = []
    
    if len(elements) > 0:
        for i in range(0,len(elements)):
            resultList1.extend([elements[i].childNodes[0].nodeValue])
    return resultList1

def resumir(texto,lenguaje='spanish',ratio=0.2):
    if not(lenguaje):
        return summarizer.summarize(texto, language='spanish',ratio=ratio)
    else:
        return summarizer.summarize(texto, language=lenguaje,ratio=ratio)    

def transformaMaterias(materiasF):
    label=[]
    materiasF=materiasF[0].split(",")
    for i in range (1,len(materiasF)):  
    
        nuevo=str(materiasF[i])
        #nuevo=self.Replace(nuevo)
        label.append(nuevo)
    
    return label 
    
def tokenize(resultList1):
    entrada=[]
    tokens = word_tokenize(resultList1)
    filtered_words = [w for w in tokens if not w in stopwords.words('spanish')]


    for i in filtered_words:
        stri = unicode(i,errors='replace')
        entrada.append(stri)

    return entrada 
    
def PreparaMaterias(materias):
    materiasTrain = []
    for i in materias:
        if len(i)>0:
            i=transformaMaterias(i)
            materiasTrain.append(i)
    
    return materiasTrain
        
"""
Ventana de la aplicación que se encarga de entrenar un modelo
"""        
class VentanaEntrenado(wx.Frame):
    
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, wx.NewId(),"Ventana de clasificacion", size=(580,350))
        self.Bind(wx.EVT_CLOSE, self.cerrar)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetBackgroundColour("white")
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.img = wx.EmptyImage(100,100)
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, 
                                         wx.BitmapFromImage(self.img))
        self.imageCtrl.SetPosition((85,100))
        self.img = wx.Image("images.jpg", wx.BITMAP_TYPE_ANY)

 
        self.imageCtrl.SetBitmap(wx.BitmapFromImage(self.img))

        self.botonElegirDirectorioClasificacion = wx.Button(self, wx.NewId(),"Elegir directorio Clasificacion",pos=(115,300))

        self.botonSalir = wx.Button(self,wx.NewId(),"Cerrar",pos=(475,300))
        self.botonClasificar = wx.Button(self,wx.NewId(),"Clasificar",pos=(375,300))
        self.botonElegirModelo = wx.Button(self,wx.NewId(), "Elegir Modelo",pos=(3,300))


        self.botonElegirModelo.Bind(wx.EVT_BUTTON,self.ClickElegirModelo)
        self.botonElegirDirectorioClasificacion.Bind(wx.EVT_BUTTON, self.ClickElegirDirectorioClasificacion)
        self.botonSalir.Bind(wx.EVT_BUTTON, self.botonClickSalir)
        self.botonClasificar.Bind(wx.EVT_BUTTON, self.ClickBotonClasificar)

        self.Layout()
        
        

    """
    Función del boton clasificar encargada de realizar la clasificación.
    """
    def ClickBotonClasificar(self, evt):
        self.MakeModal(True)
        global _resultados

        extractoT=[]
        materiasE=[]
        resumenes =[]
        extractoTs = []
        f = open(str(_selectedDirModelo))
        classifier =pickle.loads(f.read())
        f.close()

        print "EL ARCHIVO HA SIDO LEIDO"
        ficherosT = os.listdir(_selectedDirClasificacion) # linux
        
        pathsTest = []
        for i in ficherosT:
            path=_selectedDirClasificacion+"/"+i
            pathsTest.append(path)
           # materiasT.append(leerTags(path,'materias'))
           
            extracto = leerTags(path,'extracto')
            extrac = str(extracto)
            extractoTs.append(extrac)
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
                resumenes.append(final)
                todo = extrac+ " " + final
                extractoT.append(todo)
            except ValueError:
                print "error el archivo: "+ path +"No se puede resumir probablemante porque este vacio. "  
                
        ficherosE = os.listdir(_selectedDirEntrenamiento)
        
        for i in ficherosE:
            path=_selectedDirEntrenamiento+"/"+i
            materiasE.append(leerTags(path,'materias'))
           
        y_train = PreparaMaterias(materiasE)                 
        lb = preprocessing.MultiLabelBinarizer()        
        y_train = lb.fit_transform(y_train)
        print "Leidas las inicivas y preparado el test"
        ############################################################################
       
        iniciativasTest= np.array(extractoT)  
  
        probabilidad = classifier.predict_proba(iniciativasTest)
        
        print len (iniciativasTest)
        contenido=Resultados()
        contenidoRestante = Resultados()
        contenidoEtiRestantes=[]
        for i in range(0,len(iniciativasTest)):
            
            auxEti = []
            auxProba = []
            auxEtiqMenos = []
            auxProbaMenos = []
            for j in range (0, len(list(classifier.classes_))):
                label= lb.classes_[j]
                #nlabel= classifier2.classes_[j]
                prob=probabilidad[i][j]
                if prob > 0.027:             
                    auxProba.append(str(prob))
                    auxEti.append(str(label))
                if prob < 0.027:
                    auxEtiqMenos.append(str(label))
                    auxProbaMenos.append(str(prob))
                    
            contenido.etiquetas.append(auxEti) # Etiquetas preseleccionadas con mas probabilidad de salir
            contenido.probabilidades.append(auxProba)
            contenidoRestante.etiquetas.append(auxEtiqMenos)
            contenidoRestante.probabilidades.append(auxProbaMenos)


            contenidoEtiRestantes.append(auxEtiqMenos) # El resto de las seleccionadas con menos probabilidad de salir.
        print "FIN"
        _resultados = contenido
        
        #len contenido .etiquetas es el numero de iniciativas predecidas
        for i in range(0,len(contenido.etiquetas)):
            
             root = Tk()
             root.title("Ventana de elecciones")
             root.geometry("1000x1200")
             ventana = Frame(root,background="#DEDFD7")   
             ventana.pack()      
             etiquetasElegidas = []   
             def Guardar(): 
                  lista= (list(lng.state()))
                  for j in range (0,len(lista)):
                       if lista[j]==1:
                            etiquetasElegidas.append( contenido.etiquetas[i][j])
                  listaString= (str(etiquetasElegidas[0])) 
                  if len(etiquetasElegidas)>1:
                      for k in range (1,len(etiquetasElegidas)):
                          listaString= listaString + ","+(str(etiquetasElegidas[k]))
                  print "materias a guardar"
                  print listaString
                  xmlFile = parse(pathsTest[i])  
                  tagMaterias = xmlFile.createElement("materias")
                  print "creado elemento materias"
                  tagMaterias.setAttribute("materias"  , listaString)
                  xmlFile.childNodes[0].appendChild( tagMaterias )
                  with open(pathsTest[i],'w') as f:
                      f.write(xmlFile.toprettyxml())                    
                  
                  
             def visualizarIniciativa():
                   call("firefox"+" "+str(pathsTest[i]), shell=True)    
            
             
             labelText= Label(ventana,text="Resumen",font=("Helvetica", 16),fg="black" ,background="#DEDFD7")
             labelText.pack(padx=0, pady=10,side=TOP)
             
             text1=Text(ventana, height=3, width=150)
             text2 = Text(ventana, height=15, width=150)
             text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))

             text1.tag_configure('big',foreground='#A69562', font=('Verdana', 13, 'bold'),background="#F8F3E4")
             text2.tag_configure('color', foreground='black', background="#F8F3E4",font=('Tempus Sans ITC', 12, 'bold'))
             
             text1.insert(END,str(extractoTs[i]), 'big')
             text2.insert(END, resumenes[i], 'color')
             text1.pack(side=TOP,expand=False)
             text2.pack(side=TOP,expand=False)
             labelList1 = Label(ventana, text ="Elecciones con alta probabilidad de ser elegidas",font=("Helvetica", 16),fg="black",background="#DEDFD7")
             labelList1.pack(padx=0, pady=10,side=TOP)
             ConOrdenado = {}
             ResOrdenado = {}            
             for k in range(0, len (contenido.etiquetas[i])):                 
                 ConOrdenado[str(contenido.probabilidades[i][k])] = contenido.etiquetas[i][k]
             for k in range(0, len(contenidoRestante.etiquetas[i])):
                 ResOrdenado[str(contenidoRestante.probabilidades[i][k])]= contenidoRestante.etiquetas[i][k]
             

             mas = ConOrdenado.items()
             menos = ResOrdenado.items()
             mas.sort()
             menos.sort()
             mas.reverse()
             menos.reverse()
             


             eleccionesBuenas = []
             for t in range(0,len(ConOrdenado.items())):
                 eleccionesBuenas.append(mas[t][1])
                 
             eleccionesMenosBuenas = []
             for t in range(0,len(ResOrdenado.items())):
                 eleccionesMenosBuenas.append(menos[t][1])
                         
             lng = Checkbar(ventana,eleccionesBuenas)
             lng.pack(side=TOP, fill=X)
             lng.config(relief=GROOVE, bd=2)
             labelList2 = Label(ventana, text = "Resto de posibles materias con que etiquetar la iniciativa",font=("Helvetica", 16),fg="black",background="#DEDFD7")
             labelList2.pack(padx=0, pady=10,side=TOP)

             lng2 = CheckBarListScroll(ventana,eleccionesMenosBuenas)
             lng2.pack(side="top", fill="x", expand=False)
             lng2.config(relief=GROOVE, bd=2)
             Button(ventana,text='Visualizar Iniciativa',background="#C1C4B5", command= visualizarIniciativa).pack(side=RIGHT)
             Button(ventana, text='Guardar' ,background="#C1C4B5",command=Guardar).pack(side=RIGHT)
             if i == len(iniciativasTest)-1:
                 Button(ventana, text='Cerrar',background="#C1C4B5", command=root.destroy).pack(side=RIGHT) 
             else:             
                  Button(ventana, text='Siguiente', background="#C1C4B5",command=root.destroy).pack(side=RIGHT)
             root.mainloop()

              
    def ClickElegirDirectorioClasificacion(self, evt):
        
        global _selectedDirClasificacion 

        
        userPath = ''
        app = wx.App()
        dialog = wx.DirDialog(None, "Por favor elija un directorio para clasificar:",\
            style=1 ,defaultPath=userPath, pos = (10,10))
        if dialog.ShowModal() == wx.ID_OK:
            _selectedDirClasificacion = dialog.GetPath()
            print _selectedDirClasificacion
            pathModelo = ""
            for i in range(1, len(_selectedDirClasificacion.split("/"))-1):
                pathModelo = pathModelo +"/"+_selectedDirClasificacion.split("/")[i]
            pathModelo = pathModelo+"/modeloClases.txt"
            global _selectedDirEntrenamiento
            print pathModelo
            f = open(str(pathModelo),"r")
            
            _selectedDirEntrenamiento = f.readline()
            f.close()
            _selectedDirEntrenamiento = _selectedDirEntrenamiento.strip()            
                
            return _selectedDirClasificacion
        else:
            app.Close()
            
    def ClickElegirModelo(self, evt):
        
        global _selectedDirModelo 
        
        app = wx.App()
        dialog = wx.FileDialog(None, "Por favor eliga un directorio para clasificar:",\
            style=1 , pos = (10,10))
        if dialog.ShowModal() == wx.ID_OK:
            _selectedDirModelo = dialog.GetPath()
            print _selectedDirModelo
            return _selectedDirModelo
        else:
            app.Destroy()    
   
    def cerrar(self, evt):
          
        self.MakeModal(False)
        evt.Skip()  
        
    def botonClickSalir(self, evt):
        self.MakeModal(False)
        self.Close()

"""
Clase checkbar, para poder crear listas de checkbotton

"""
class Checkbar(Frame):
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
      Frame.__init__(self, parent)
      self.root = parent
      self.text =Text(self, width=4, height=3,background="#F8F3E4")
      self.text.pack(side="left", fill="x", expand=True)
      self.vars = []
      for pick in picks:
         var = IntVar()
         cb = Checkbutton(self, text= pick,variable=var,background="#F8F3E4")
         self.text.window_create("end", window=cb)
         self.text.insert("end"," ") 
         self.vars.append(var)
   def state(self):
      return map((lambda var: var.get()), self.vars)
      
"""
Clase para poder crear una lista de checkbotton en la que sea posible
desplazarnos hacia abajo o hacia arriba.
"""
class CheckBarListScroll(Frame):
    def __init__(self,parent=None, picks=[], side=LEFT, anchor=W):
        Frame.__init__(self, parent)
        self.root = parent

        self.vsb = Scrollbar(self, orient="vertical",background="#F8F3E4")
        self.text =Text(self, width=4, height=9, 
                            yscrollcommand=self.vsb.set,background="#F8F3E4")
        self.vsb.config(command=self.text.yview)
        self.vsb.pack(side="right", fill="y",expand=False)
        self.text.pack(side="left", fill="both", expand=True)
        self.vars=[]
        for i in picks:
            var = IntVar()
            cb = Checkbutton(self, text= i,variable=var,background="#F8F3E4")
            self.text.window_create("end", window=cb)
            self.text.insert("end", "\n") 
            self.vars.append(var)
    def state(self):
      return map((lambda var: var.get()), self.vars)
"""
Clase para identificar el tipo de dato Resultado
Esta clase enlazara las etiquetas con las probabilidad de cada muestra
de pertener a una clase.
"""

class Resultados():

     etiquetas = []
     probabilidades = []
     
     def __init__(self ):
         self.etiquetas = []
         self.probabilidades = []

"""
Clase clasificador.
Permite crear un clasificador y realizar las tareas de clasificacion y etiquetado.
"""
class Clasificador():

    def main(self,direccionTest,direccionEntrenamiento):

        ficherosT = os.listdir(direccionTest) # linux
        ficherosE = os.listdir(direccionEntrenamiento)
        materiasE = []
        extractoE = []
        materiasT = []
        extractoT = []

        for i in ficherosE:
            path=direccionEntrenamiento+"/"+i
            materiasE.append(self.leerTags(path,'materias'))
            
            extracto = self.leerTags(path,'extracto')
            extr = str(extracto)
            parrafos = self.leerTags(path,'parrafo')
            todo=extr
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
        ############################################################################
        for i in ficherosT:
            path=direccionEntrenamiento+"/"+i
            materiasT.append(self.leerTags(path,'materias'))
           
            extracto = self.leerTags(path,'extracto')
            extr = str(extracto)
            
            parrafos = self.leerTags(path,'parrafo')
            todo=extr
            for i in range(0,len(parrafos)):
                todo = todo +" "+str(parrafos[i])  
            resumenes.append(todo)
            try:

                todo = resumir(todo,ratio=0.25)
                todo= tokenize(todo) 
            
                final = ""    
                for i in range(0, len (todo)):
                    final = final+" "+str(todo[i])    
                final =str(final) 
                extractoT.append(final)
                
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
        ficheros = os.listdir(direccionEntrenamiento) # linux
        materias = []
        # Extraccion de las materias por cada iniciativa del directorio
        # las termino uniendo en una lista.
        for i in ficheros:
            path=str(direccionEntrenamiento)+"/"+i
            midom=parse(path)    
            elements = midom.getElementsByTagName('materias')
            resultList = []
            if len(elements) != 0:
                for i in range(0,len(elements)):
                    resultList.extend([elements[i].childNodes[0].nodeValue])
                    materias.append(resultList[i])
        
        target_names=self.GeneraTarget(materias)          

        return self.clasificador(iniciativasTraining, y_train, iniciativasTest, target_names, y_test,target_names)        
  
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
    def resumir(self,texto,lenguaje='spanish',ratio=0.25):
        if not(lenguaje):
            return summarizer.summarize(texto, language='spanish',ratio=ratio)
        else:
            return summarizer.summarize(texto, language=lenguaje)

    ############################################################################
    ############################################################################  
    
    def entrenar (self,direccionEntrenamiento):
       
        ficherosE = os.listdir(direccionEntrenamiento)
        materiasE = []
        extractoE = []


#        resumenes = []
        # Leemos los de entrenamiento
        print "LEYENDO INICIATIVAS"
        for i in ficherosE:
            path=direccionEntrenamiento+"/"+i
            materiasE.append(self.leerTags(path,'materias'))
            
            extracto = self.leerTags(path,'extracto')
            extrac= str(extracto)##############################
            #extractoE.append(todo)
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
                final =str(final)##remune#######################################
                #resumenes.append(final)
           
            except ValueError:
                print "error el archivo: "+ path +"No se puede resumir probablemante porque este vacio. " 
            
            aux=extrac + final
            extractoE.append(aux)
        print "FIN DE LA LECTURA DE INICIATIVAS"  
            
        ############################################################################

        ############################################################################
        #y_test = PreparaMaterias(materiasT)
   
        y_train = self.PreparaMaterias(materiasE)
        X_train = np.array(extractoE)
        
        
        
        lb = preprocessing.MultiLabelBinarizer()

        y_train = lb.fit_transform(y_train)
        
        classifier = Pipeline([
            ('vectorizer',CountVectorizer(strip_accents='unicode')),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(SVC(kernel='linear',C=1.0, cache_size=200, coef0=0.0,
                                               gamma=0.0, max_iter=-1,  random_state=None,
                                               shrinking=True, tol=0.001, verbose=False,probability=True)))])
        
        print "ENTRENANDO"
        classifier.fit(X_train,y_train)
        # se crea el modelo y se almacena en una variable
        print "Modelo creado"
        s = pickle.dumps(classifier)

        return s
        
    def clasificador(self,X_train, y_train, X_test, target_names, y_test,all_labels):
        
        lb = preprocessing.MultiLabelBinarizer()

        y_train = lb.fit_transform(y_train)
        
        classifier = Pipeline([
            ('vectorizer',CountVectorizer(strip_accents='unicode')),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(SVC(kernel='linear',C=1.0, cache_size=200, coef0=0.0,
                                               gamma=0.0, max_iter=-1,  random_state=None,
                                               shrinking=True, tol=0.001, verbose=False,probability=True)))])
                    
        classifier.fit(X_train,y_train)
        # se crea el modelo y se almacena en una variable
        s = pickle.dumps(classifier)
        self.modelo = s
      
        #####################################################################################################################################

        probabilidad = classifier.predict_proba(X_test)

        contenido=Resultados()
        for i in range(0,len(probabilidad)):
            
            auxEti = []
            auxProba = []

            for j in range (0, len(list(lb.classes_))):
                label= lb.classes_[j]
                #nlabel= classifier2.classes_[j]
                prob=probabilidad[i][j]
                if prob > 0.027:             
                    auxProba.append(str(prob))
                    auxEti.append(str(label))
            
            contenido.etiquetas.append(auxEti)
            contenido.probabilidades.append(auxProba)
      
        print "FIN"
        return contenido           
         
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
"""
Clase principal de la aplicación.
"""
class MiApp(wx.App):
   
   def OnInit(self):
       ventana = VentanaPrincipal()
       ventana.Show(True)
       return True
       
app = MiApp(0)
app.MainLoop()


