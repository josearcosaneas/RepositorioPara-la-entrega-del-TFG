#!/bin/bash

# -*- ENCODING: UTF-8 -*-
#
# script de instalacion de setuptools y dependencias
#
echo "Instalando dependendicas de python"
echo "Necesitara estar logado como root"

sudo apt-get update

sudo apt-get install python-setuptools python-dev build-essential python-wxgtk2.8 python-wxtools wx2.8-doc wx2.8-examples wx2.8-headers wx2.8-i18n python-tk python-pip python-numpy python-matplotlib ipython python-pandas python-sympy python-nose python-scipy
sudo pip install -U numpy
sudo pip install networkx
#llamada al archivo setup.py la instalacion del resto de dependencias.
sudo python setup.py install

cd GuiClasificador.egg-info
# Aquí hacemos que pypi realiza la instalación de los requisitos de
# dependencias generados por setup.py
sudo pip install -r requires.txt

echo " Dependencias instaladas "
