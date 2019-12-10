#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("../Datasets/College.csv")
df.head()


# In[4]:


df = df.drop(['Unnamed: 0'],axis=1)
df.info()


# In[5]:


df.describe()


# In[6]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[7]:


df['Private'] = df['Private'].apply(converter)
df.info()


# In[9]:


sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df,
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[10]:


sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df,
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[17]:


from sklearn.cluster import KMeans
X = df.drop(['Private'],axis=1)
y = df["Private"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

kmeans = KMeans(n_clusters=2, random_state = 10)
kmeans.fit(X_train)


# In[18]:


predicted = kmeans.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predicted))


# In[19]:


cm = confusion_matrix(y_test,predicted)
sns.heatmap(cm, center=True)
plt.show()


# In[20]:


#Plotando os valores do cluster do "fit"
df_clusters_train = pd.DataFrame()
df_clusters_train.loc[:,"cluster"] = pd.Series(kmeans.labels_)
df_clusters_train = pd.concat([df_clusters_train,X_train],axis=1)

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df_clusters_train,hue="cluster",
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[21]:


#Plotando os valores do cluster da predição
df_clusters_test = pd.DataFrame()
df_clusters_test.loc[:,"cluster"] = pd.Series(predicted)
df_clusters_test = pd.concat([df_clusters_test,X_test],axis=1)

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df_clusters_test,hue="cluster",
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[22]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(X)

X = pd.DataFrame(data=principalComponents)
X = X.values


# In[23]:


kmeans_colors = kmeans.fit_predict(X)
C = kmeans.cluster_centers_


# In[40]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans_colors)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='o', c='#FF4500', s=1000)


# In[41]:


plt.show()


# In[ ]:




