# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:24:42 2021

@author: CHANDRALEKHA
"""

import pandas as pd

df = pd.read_csv('titanic.csv')
df.info()

df["age"] = pd.to_numeric(df['age'], errors='coerce')
df["fare"] = pd.to_numeric(df['fare'], errors='coerce')
df.info()
df.isna().sum()

df["age"].fillna(df["age"].mean(), inplace=True)
df.dropna(subset=['fare'],inplace=True)
df.isnull().sum()

s_dummy = pd.get_dummies(df["sex"],drop_first=True)

df['embarked'].describe()
df['embarked']=df['embarked'].replace(({'?': 'S'}))

emb_dummy = pd.get_dummies(df["embarked"],drop_first=True)


p_dummy = pd.get_dummies(df["pclass"],drop_first=True)

df = pd.concat([df,s_dummy,p_dummy,emb_dummy],axis = 1)
df.info()

df.drop(["sex","embarked","pclass","Passenger_id","name","ticket"],axis=1,inplace=True)

x=df.iloc[:,[1,2,4,5,7,9]]
y=df["survived"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
decision_mod =DecisionTreeClassifier(criterion='entropy',max_depth =3,min_samples_leaf=5)

decision_mod.fit(x_train,y_train)

y_pred = decision_mod.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

#hence , accuracy = (171+87)/(171+87+41+24) = 79.87%

from sklearn import tree
tree.plot_tree(decision_mod)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
cn=['0','1']
tree.plot_tree(decision_mod,class_names=cn,filled = True)
