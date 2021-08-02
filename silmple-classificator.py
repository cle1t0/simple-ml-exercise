#The original dataset was edited, adding the first rows as label in order to make the feature/output division easier
#Kernel tricks were used because of the non-linear dicision boundary

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#Reading the archive
data =  pd.read_csv('vetsRots.txt',sep=' ',header=0) 
#print(data.head)

#Setting the 'class' column as the output
x = data.drop('class',axis=1)
y = data['class']

#Here I ask the user to define a percentage for the train-test split
val = input("Insira a porcentagem de treino : ") 
val =  int(val)
print('O treino será de',val,'%')
valtreino  = (val/100)

#Trani-test split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = valtreino)

#Training the model using the Gaussian Kernel method
maquina = SVC(kernel='rbf')
maquina.fit(x_treino, y_treino)

#Predictions and accuracy response
y_prever = maquina.predict(x_teste)
print(classification_report(y_teste, y_prever))

