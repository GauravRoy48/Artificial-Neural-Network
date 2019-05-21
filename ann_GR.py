#####################################################################################
# Creator     : Gaurav Roy
# Date        : 21 May 2019
# Description : The code performs Artificial Neural Network algorithm on the 
#               Churn_Modelling.csv. 
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

# Categorical variables present as there are strings in X
# Encoding X categorical data + HotEncoding

# Encoding for the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

# Encoding for the Geography Column 
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding Dummy Variable Trap
X = X[:,1:]

# Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting ANN to Training Set
# Create ANN Here
# Importing Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding Input Layer and First Hidden Layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Add Second Hidden Layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Add Output Layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

###############################################################################
# RUN SEPARATELY

# Fitting ANN to training set
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)
###############################################################################

# Making Predictions and Evaluating Model

# Predicting the Test Set Results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)