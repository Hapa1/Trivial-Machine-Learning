#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Upscaling code taken from https://elitedatascience.com/imbalanced-classes

df = pd.DataFrame(pd.read_csv("C:/Users/Steve/Desktop/Code/indian_liver_patient.csv"))
df = df[np.isfinite(df['Albumin_and_Globulin_Ratio'])]
for index, row in df.iterrows():
    if row['Gender'] == 'Male':
        df.at[index, "Gender"] = 0
    if row['Gender'] == 'Female':
        df.at[index, "Gender"] = 1 
        
df['Gender'] = df['Gender'].astype('int64')

df_majority = df[df.Dataset==1]
df_minority = df[df.Dataset==2]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=414,  # to match majority class
                                 random_state=999) # reproducible results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.Dataset.value_counts()

y = df_upsampled.Dataset
#y=y.astype('int') #convert Column 'Class' type to number from object 
X = df_upsampled.drop('Dataset', axis=1)

#X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999) #Split Data
clf_1 = KNeighborsClassifier(n_neighbors=5, p = 1) #5-NN Using Manhattan Distance 
clf_1 = clf_1.fit(X_train, y_train) # Train model
pred_y_1 = clf_1.predict(X_test) # Predict on testing set
print( accuracy_score(y_test, pred_y_1) ) # Print accuracy

clf_2 = KNeighborsClassifier(n_neighbors=5, p = 2) #5-NN Using Euclidean Distance (Default)
clf_2 = clf_2.fit(X_train, y_train) # Train model
pred_y_2 = clf_2.predict(X_test) # Predict on testing set
print( accuracy_score(y_test, pred_y_2) ) # Print accuracy

clf_3 = KNeighborsClassifier(n_neighbors=5, metric = 'chebyshev') #5-NN Using Chebyshev Distance (Default)
clf_3 = clf_3.fit(X_train, y_train) # Train model
pred_y_3 = clf_3.predict(X_test) # Predict on testing set
print( accuracy_score(y_test, pred_y_3) ) # Print accuracy


# In[ ]:




