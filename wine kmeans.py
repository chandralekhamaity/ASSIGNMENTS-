# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:22:06 2021

@author: CHANDRALEKHA
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('wine.csv')
X = df.iloc[:,[0,9]].values

from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters= i ,init= 'k-means++' , random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('wine')
plt.ylabel('WCSS')
plt.savefig('output.png')

kmeans = KMeans(n_clusters = 5 , init = 'k-means++' , random_state = 42 )
y_kmeans = kmeans.fit_predict(X)

kmeans = pd.DataFrame(y_kmeans)
df_1 = pd.concat([df , kmeans], axis = 1)

plt.scatter(X[y_kmeans ==0,0], X[y_kmeans ==0,1], s = 100 , c = 'indianred' , label = 'Cluster1')
plt.scatter(X[y_kmeans ==1,0], X[y_kmeans ==1,1], s = 100 , c = 'coral' , label = 'Cluster2')
plt.scatter(X[y_kmeans ==2,0], X[y_kmeans ==2,1], s = 100 , c = 'khaki' , label = 'Cluster3')
plt.scatter(X[y_kmeans ==3,0], X[y_kmeans ==3,1], s = 100 , c = 'olivedrab' , label = 'Cluster4')
plt.scatter(X[y_kmeans ==4,0], X[y_kmeans ==4,1], s = 100 , c = 'rosybrown' , label = 'Cluster5')

plt.title('Cluster of wine')
plt.xlabel('Alcohol')
plt.ylabel('Color intensity')
plt.legend()
plt.show()
