#####################################################################################
# Creator     : Gaurav Roy
# Date        : 23 May 2019
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
from keras.layers import Dropout # To avoid overfitting

# Initializing the ANN
classifier = Sequential()

# Adding Input Layer and First Hidden Layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(rate = 0.1))

# Add Second Hidden Layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate = 0.1))


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


# Testing on new entry
'''
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''
new_pred = classifier.predict(sc_X.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5) # False - So customer will not leave


## Evaluating ANN
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#
#def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
#    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#    return classifier
#
#classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
#accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
#avg_accuracies = accuracies.mean()
#std_accuracies = accuracies.std()

# Tuning ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, Y_train)
# Gives the accuracy with the best parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
