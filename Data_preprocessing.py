# Data preprocessing

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset

dataset = pd.read_csv('data.csv')

# description

dataset.describe()

# variables d'entrée varibale cible

X = dataset.iloc[:,:-1].values  # matrice
y = dataset.iloc[:,3].values    # vecteur

# Taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True)

# Encoding categorical data
# Encoding the Independent Variable

imputer = imputer.fit(X[:,1:3]) # on n'a pas encore transformé les colonnes , donc pas de changements
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder , OneHotEncoder # on veut remplacer les noms de pays par des chiffres
labelEncoder_X =  LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable

labelEncoder_y =  LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                  test_size=0.2,\
                                                  random_state=0)
y_train = y_train.reshape(-1,1)
# Feature Scaling : mise à l'echelle des caractériques des variables

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # changer et transformer sauvegarder
X_test = sc_X.transform(X_test)     

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)




