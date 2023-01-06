#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv("student.csv")


# In[4]:


df


# In[6]:


df.drop_duplicates(subset=["name"],keep="last",inplace=True)
df.head(10)
df.isnull().sum()


# In[10]:


df_dropped=df.dropna()


# In[11]:


df.head()


# In[14]:


df['Address']=df['Address'].fillna('koriyar')
df.head()


# In[16]:


mean_val=df['MOBILE'].mean()
mean_val=df['MOBILE'].fillna(mean_val)
df


# In[22]:


import pandas as pd
import numpy as np
array=np.array([[1,1,1],[2,3,8],[3,5,27]])
index_values=['Row1','Row2','Row3']
column_values=['Col1','Col2','Col3']
df=pd.DataFrame(data=array,
               index=index_values,
               columns=column_values)
print(df)
trace=np.trace(df)
print("\n Trace of given 3*3 matrix")
print(trace)


# In[27]:


#array manipulation,Searching,Sorting and spliting
import numpy as np
lt=[]
n=int(input("Enter number of elements:"))
for i in range(0,n):
    ele=int(input())
    lt.append(ele)
print(lt)
arr=np.array(lt)
arr=np.sort(arr)
print("Sorted Array={}".format(arr))
ele=int(input("enter element to seach"))
i=np.where(arr==ele)
print("index={}".format(i))
s=int(input("Enter spliting values:"))
newarr=np.array_split(arr,s)
print(newarr)


# In[28]:


#array manipulation,Searching,Sorting and spliting
import numpy as np
lt=[]
n=int(input("Enter number of elements:"))
for i in range(0,n):
    ele=int(input())
    lt.append(ele)
print(lt)
arr=np.array(lt)
arr=np.sort(arr)
print("Sorted Array={}".format(arr))
ele=int(input("enter element to seach"))
i=np.where(arr==ele)
print("index",i)
s=int(input("Enter spliting values:"))
newarr=np.array_split(arr,s)
print(newarr)


# In[34]:


#broadcasting
import numpy as np
from numpy import array,argmin,sqrt,sum
observation=array([111.0,188.0])
codes=array([[102.0,203.0],
           [1312.0,193.0],
           [57.0,173.0]])
diff=codes-observation
dist=sqrt(sum(diff**2,axis=-1))
print(observation,observation.shape)
print(codes,codes.shape)
print(diff,diff.shape)
print(dist,dist.shape)
argmin(dist)


# In[41]:


import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,3*np.pi,0.1)
y_sin=np.sin(x)
y_cos=np.cos(x)
plt.plot(x,y_sin)
plt.plot(x,y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine','Cosine'])
plt.show()


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
data={'year':[1920,1920,1940,1950,1960,1970,1980,1990,2000,2010],
     'unemployment_rate':[9,8,12,8,7,2,6.9,7,6.5,6.2]}
df=pd.DataFrame(data)
plt.plot(df['year'],df['unemployment_rate'],color='red',marker='p')
plt.title('unemployment rate vs year',fontsize=14)
plt.xlabel('unemployment rate',fontsize=14)
plt.xlabel('year',fontsize=14)
plt.ylabel('unemployment rate',fontsize=14)
plt.grid(True)
plt.show()


# In[15]:


#Scatter plot
import matplotlib.pyplot as plt
import numpy as np
x=np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y=np.array([99,86,87,88,111,86,103,86,7,94,78,77,85])
plt.scatter(x,y)
plt.show()


# In[53]:


#Scatter plot
import matplotlib.pyplot as plt
import numpy as np
x=np.array(["A","B","C","D"])
y=np.array([3,8,1,10])
plt.bar(x,y)
plt.show()


# In[55]:


#Scatter plot
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
dataSet1=np.random.normal(100,10,200)
dataSet2=np.random.normal(80,20,200)
dataSet3=np.random.normal(60,35,220)
dataSet4=np.random.normal(50,40,200)
dataSet=[dataSet1,dataSet2,dataSet3,dataSet4]
figure=plt.figure(figsize=(10,7))
ax=figure.add_axes([0,0,1,1])
bp=ax.boxplot(dataSet)
plt.show()


# In[3]:


#histogram
from matplotlib import pyplot as plt
import numpy as np
fig,ax=plt.subplots(1,1)
a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
ax.hist(a,bins=[0,25,50,75,100])
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel("no of students")
plt.show()


# In[8]:


#heatmaps
from pandas import DataFrame
import matplotlib.pyplot as plt
data=[{8,9,3,4},{4,6,9,1},{7,8,1,3},{9,5,1,3},{7,1,3,9}]
Index=['I1','I2','I3','I4','I5']
Cols=['C1','C2','C3','C4']
df=DataFrame(data,index=Index,columns=Cols)
plt.pcolor(df)
plt.show()


# In[6]:


#pie chart
import matplotlib.pyplot as plt
import numpy as np
y=np.array([35,25,25,15])
mylabels=["Apples","Bananas","Cherries","Dates"]
plt.pie(y,labels=mylabels,startangle=90)
plt.show()


# In[14]:


# 7 program
from matplotlib import pyplot
import random
data=[[random.randint(0,256) for x in range(0,5)],
     [random.randint(0,256) for x in range(0,5)],
     [random.randint(0,256) for x in range(0,5)],
     [random.randint(0,256) for x in range(0,5)],
     [random.randint(0,256) for x in range(0,5)],
     [random.randint(0,256) for x in range(0,5)]]
data

from matplotlib import colors
pyplot.figure(figsize=(5,5))
pyplot.xlabel("x axis with ticks",size=8)
pyplot.ylabel("y axis with ticks",size=8)
pyplot.title("this is title of the plot",size=10)
pyplot.xticks(size=14,color="red")
pyplot.yticks(size=14,color="red")
colormap=colors.ListedColormap(["lightpink","darkblue"])
pyplot.imshow(data,cmap=colormap)


# In[13]:


#prgoram 8
#Importing DataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('sal.csv')
dataset.head()
x=dataset.iloc[:,:-1].values   #independent variable array
y=dataset.iloc[:,1].values    #dependent variable array
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#fitting thne regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results
y_pred=regressor.predict(x_test)
y_pred
y_test

#plot for the train
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("salary vs experience(Training set)")

plt.xlabel("years of experience")
plt.ylabel("salaries")
plt.show()

#plot for test
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("salary vs experience(Testing set)")

plt.xlabel("years of experience")
plt.ylabel("Salaries")
plt.show()


# In[ ]:




