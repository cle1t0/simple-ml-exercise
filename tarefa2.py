#Tarefa2 -  Thiago Nishimura de Sousa
#O arquivo original txt foi editado, nomeando-se as colunas para melhor identificação das classes
#Para esse exercícios foi utilizado "kernel tricks" para que a SVM pudesse trabalhar com dados sem um fronteira linear de decisão

#Bibliotecas utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#Lendo o arquivo
data =  pd.read_csv('vetsRots.txt',sep=' ',header=0) 
#print(data.head)

#Processando os dados, para poder utilizar a coluna "class" como rótulo
x = data.drop('class',axis=1)
y = data['class']

#Pedindo ao usuario para colocar a porcentagem de dados para teste
val = input("Insira a porcentagem de treino : ") 
val =  int(val)
print('O treino será de',val,'%')
valtreino  = (val/100)

#Separando os dados entre treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = valtreino)

#Treinando a SVM com o modelo de Kernel Gaussiano
maquina = SVC(kernel='rbf')
maquina.fit(x_treino, y_treino)

#Previsão e Acurácia
y_prever = maquina.predict(x_teste)
print(classification_report(y_teste, y_prever))

