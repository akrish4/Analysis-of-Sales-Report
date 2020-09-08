# -*- coding: utf-8 -*-
"""salesreport.ipynb


IMPORT LIBRARIES
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns

"""IMPORT DATASET"""

dataset= pd.read_csv('Attribute DataSet.csv',
                  na_values="?" )
dataset.head()

"""TO SEPERATE DATA WITH DATA TYPE OBJECT  TO FILL IN NULL VALUES"""

dataset.dtypes

obj_dataset = dataset.select_dtypes(include=['object']).copy()
obj_dataset.head()aa

"""TO SEE NULL VALUES PRESENT"""

obj_dataset[obj_dataset.isnull().any(axis=1)]

"""TO DETERMINE WHICH VALUE IS REPEATED THE MOST TO FILL THE NULL VALUE WITH"""

obj_dataset["waiseline"].value_counts()

obj_dataset["Material"].value_counts()

obj_dataset["FabricType"].value_counts()

obj_dataset["Decoration"].value_counts()

obj_dataset["Pattern Type"].value_counts()

"""FILLING THE NULL VALUES WITH THE MOST REPEATED VALUE"""

obj_dataset = obj_dataset.fillna({"waiseline": "natural"})
obj_dataset = obj_dataset.fillna({"Material": "cotton"})
obj_dataset = obj_dataset.fillna({"FabricType": "chiffon"})
obj_dataset = obj_dataset.fillna({"Decoration": "lace"})
obj_dataset = obj_dataset.fillna({"Pattern Type": "solid"})
dataset.head()

"""COMPLETELY FILLED DATASET"""

obj_dataset.insert(0,'Dress_ID',dataset.iloc[:,0])
obj_dataset.insert(3,'Rating',dataset.iloc[:,3])
obj_dataset.head()

"""ENCODING CATEGORICAL DATA"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
id = le.fit_transform(obj_dataset.iloc[:,0 ])
print(id) 
style = le.fit_transform(obj_dataset.iloc[:,1])
print(style)
rating = le.fit_transform(obj_dataset.iloc[:,3])
print(rating)
size = le.fit_transform(obj_dataset.iloc[:,4])
print(size)
waiseline = le.fit_transform(obj_dataset.iloc[:,8])
print(waiseline)
material = le.fit_transform(obj_dataset.iloc[:,9])
print(material)
fabric = le.fit_transform(obj_dataset.iloc[:,10])
print(fabric)
decoration = le.fit_transform(obj_dataset.iloc[:,11])
print(decoration)
pattern = le.fit_transform(obj_dataset.iloc[:,12])
print(pattern)

df = [id,style,rating,size,waiseline,material,fabric,decoration,pattern]
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)
X1 = pd.DataFrame(df) 
X = X1.T

"""SPLITTING INTO TESTING AND TRAINING DATA"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

print(X_train)

print(X_test)

print(y_test)

print(y_train)

"""FEATURE SCALING"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print(X_test)

print(X_train)

"""LOADING THE MODEL ON TRAINING DATA SET"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

"""1) TO PREDICT THE RECOMMENDATION OF PRODUCTS 0/1"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""ACCURACY OF THE MODEL"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

"""2) TO DETERMINE TREND OF TOTAL SALES OR EACH DRESS"""

df= pd.read_csv('Dress Sales.csv')
Z = df.iloc[:,].values

print(Z)
print(y)

"""TO FILL NULL VALUES WITH 0"""

dress = df.iloc[:,0]
df = df.fillna(0)
df.head()

"""TO FIND TOTAL SALE VALUES FOR EACH DRESS"""

df.drop('Dress_ID',axis=1)
a= df.sum(axis= 1, skipna = True) 
print(a)

a.plot(kind="bar", figsize=(18,10), stacked=True)

"""3) TO FIND HOW STYLE,SEASON AND MATERIAL EFFECT SALES

STYLE:
"""

dataset['Style'] = dataset['Style'].str.title() 
stylesale = dataset.groupby(['Style','Recommendation']).size().unstack().fillna(0)
stylesale

e = stylesale.sort_values(1,axis = 0,ascending=False)
e.plot(kind="bar", figsize=(12,8), stacked=True)

"""SEASON:"""

dataset['Season'] = dataset['Season'].str.title() 
seasonsale = dataset.groupby(['Season','Recommendation']).size().unstack().fillna(0)
seasonsale

f = seasonsale.sort_values(1,axis = 0,ascending=False)
f.plot(kind="bar", figsize=(12,8), stacked=True)

"""MATERIAL:"""

dataset['Material'] = dataset['Material'].str.title() 
matsale = dataset.groupby(['Material','Recommendation']).size().unstack().fillna(0)
matsale

d = matsale.sort_values(1,axis = 0,ascending=False)
d.plot(kind="bar", figsize=(12,8), stacked=True)

dataset['Price'] = dataset['Price'].str.title() 
pricesale = dataset.groupby(['Price','Recommendation']).size().unstack().fillna(0)
pricesale

g = pricesale.sort_values(1,axis = 0,ascending=False)
g.plot(kind="bar", figsize=(12,8), stacked=True)

"""TO DETECT IF STYLE IS MORE INFLUENTIAL THAN PRICE"""

dataset[dataset.Recommendation==1].count()

"""4)To determine which attribute makes how much impact for sales"""

#count when recommendation=1 of the factor/total count of recommendation=1
stylecontribution = 210/210*100
print('stylecontribution= ',stylecontribution)
pricecontribution = 208/210*100
print('pricecontribution= ',pricecontribution)
ratingcontribution = 210/210*100
print('ratingcontribution= ',ratingcontribution)
sizecontribution = 210/210*100
print('sizecontribution= ',sizecontribution)
seasoncontribution = 209/210*100
print('seasoncontribution= ',seasoncontribution)
necklinecontribution = 208/210*100
print('necklinecontribution= ',necklinecontribution)
sleevelengthcontribution = 209/210*100
print('sleevelengthcontribution= ',sleevelengthcontribution)
waiselinecontribution = 170/210*100
print('waiselinecontribution= ',waiselinecontribution)
materialcontribution = 142/210*100
print('materialcontribution= ',materialcontribution)
fabrictypecontribution = 104/210*100
print('fabrictypecontribution= ',fabrictypecontribution)
decorationcontribution = 112/210*100
print('decorationcontribution= ',decorationcontribution)
patterntypecontribution = 148/210*100
print('patterntypecontribution= ',patterntypecontribution)

"""5)To check if rating effects sales"""

ratingcontribution = 210/210*100
print('ratingcontribution= ',ratingcontribution)

"""#Yes,rating effects sales"""
