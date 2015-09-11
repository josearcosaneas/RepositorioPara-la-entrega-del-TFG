# -*- coding: utf-8 -*-
"""
@author: Jose AA
@author_email: joseaa@correo.ugr.es

Archivo de configuración e instalación de dependencias.
La ejecución de este archivo permite la instalación de las dependencias necesarias para 
poder usar la aplicación desarrollada.


"""

from setuptools import setup

setup(
    name = "GuiAplicacion",
    version = "0.4",
    
    scripts = ['Aplicacion.py'],
    install_requires = ['nltk == 3.0.2',
                        'scikit-learn == 0.16.1',
                        'wxPython',
                        'summa == 0.0.7',
                        'pickle-converter'],
    # dependencias que no estan en pypi
    dependency_links=['https://github.com/enthought/Python-2.7.3/blob/master/Lib/pickle.py',
                        'https://pypi.python.org/pypi/wxPython/2.8.11.0',
                        'https://pypi.python.org/pypi/scikit-learn/0.16.1'],

    zip_safe = False,
    author = "joseaa",
    author_email = "joseaa@correo.ugr.es",
    description = "Clasificador multietiqueta",
    license = "MIT",
    keywords = "SVM, Clasificador, Texto, Multilabel",
    url = "",   
)
