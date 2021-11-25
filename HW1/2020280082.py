# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def set_up_1a(filtered_df):
    avg_year=np.array(filtered_df['Average year'].values.tolist()).reshape(-1,1)
    no_response_ratio = filtered_df['No-response conversation ratio'].values.tolist()
    night_conv_ratio = filtered_df['Night conversation ratio'].values.tolist()
    img_ratio = filtered_df['Images ratio'].values.tolist()
    y=[]
    y.append(no_response_ratio)
    y.append(night_conv_ratio)
    y.append(img_ratio)
    return y, avg_year

def error_variance(y, predicted_y):
    error_list = []
    for i in range (len(y)):
        error_list.append(abs(y - predicted_y))
    print("Error variance is: ", np.var(error_list))
    
def plot_results(y, avg_year):
    y_label=['No response ratio','Night conversation ratio','Images ratio']
    for i in range(len(y)):
        np_y=np.array(y[i])
        model = LinearRegression().fit(avg_year,np_y)
        predicted_y= model.predict(avg_year) 
        
        ##plotting
        plt.scatter(avg_year,np_y, s=10)
        plt.xlabel('Average year')
        plt.ylabel(y_label[i])
        
        # predicted values
        plt.plot(avg_year,predicted_y,color='r')
        plt.show()
        plt.close()
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        error_variance(np_y, predicted_y)
        print(avg_year.reshape(1,-1))
        print("The correlation between x and y is:\n",np.corrcoef(avg_year.reshape(1,-1),np_y))


    
    
def exercise_1a(filtered_df):
    y, avg_year = set_up_1a(filtered_df)
    plot_results(y,avg_year)
    
def multi_variate_1b(df):
    weight = np.array(df['Session number'].values.tolist())
    x= np.array(df.iloc[:,2:10].values.tolist())
    no_response_ratio = df['No-response conversation ratio'].values.tolist()
    night_conv_ratio = df['Night conversation ratio'].values.tolist()
    img_ratio = df['Images ratio'].values.tolist()
    y=[]
    y.append(no_response_ratio)
    y.append(night_conv_ratio)
    y.append(img_ratio)

    for i in range(len(y)):
        np_y = np.array(y[i])
        model = LinearRegression().fit(x,np_y,weight)
        predicted_y= model.predict(x)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        error_variance(np_y, predicted_y)
        print("\n\n")

def log_regression_1c(x, y): 
    x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=1)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    model.predict(x_test)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    print("score: ",model.score(x_test,y_test))
    print("\n\n")
    
def simple_random_sampling(df,label):
    sample_df = df.sample(204)
    x=np.array(sample_df['Average year'].values.tolist()).reshape(-1,1)
    y = np.array(sample_df[label].values.tolist()) 
    model = LinearRegression().fit(x, y)
    predicted_y = model.predict(x)
    return model.intercept_, model.coef_, y
    
def cluster_sampling(df,label):
     cluster_index = random.sample(range(1,41),4)
     start_index_list = []
     end_index_list = []
     x_cluster = []
     y_cluster = []
     for i in range(len(cluster_index)):
         end_index_list.append(cluster_index[i]*51)
         start_index_list.append(end_index_list[i]-51)
         temp_df = df.iloc[start_index_list[i]:end_index_list[i]]
         x_cluster.append(temp_df['Average year'].values.tolist())
         y_cluster.append(temp_df[label].values.tolist())
     x = np.array(x_cluster)
     y = np.array(y_cluster)
     model = LinearRegression().fit(x, y)
     predicted_y = model.predict(x)
     return model.intercept_, model.coef_,predicted_y
     
def exercise_2(df,label):
    intercept_random = []
    slope_random = []
    predicted_y_random = []
    intercept_cluster = []
    slope_cluster = []
    predicted_y_cluster = []
    for i in range(1000):
        intercept, slope, predicted_y = simple_random_sampling(df,label)
        intercept_random.append(intercept)
        slope_random.append(slope)
        predicted_y_random.append(predicted_y)
        intercept, slope, predicted_y = cluster_sampling(df,label)
        intercept_cluster.append(intercept)
        slope_cluster.append(slope)
        predicted_y_cluster.append(predicted_y)
    print("mean and variance of ",label)
    print("mean of simple intercepts",np.mean(intercept_random))
    print("mean of simple slopes",np.mean(slope_random))
    print("variance of simple intercepts",np.var(intercept_random))
    print("variance of simple slopes",np.var(slope_random))
    print("variance of simple error variance",np.var(predicted_y_random))
    print("mean of cluster intercepts", np.mean(intercept_cluster))
    print("mean of cluster slopes",np.mean(slope_cluster))
    print("Variance of cluster intercepts", np.var(intercept_cluster))
    print("variance of cluster slopes",np.var(slope_cluster))
    print("variance of cluster error variance",np.var(predicted_y_cluster))
    print("\n\n")
        
     
####
df = pd.read_excel(r'data.xlsx',usecols='G,K,L,M,N')
filtered_df= df.loc[(df['Session number']>=20)]
exercise_1a(filtered_df)
df = pd.read_excel(r'data.xlsx')
#multi_variate_1b(df)


filtered_df = df[(df['Group category']==1) | (df['Group category']==4)]
y = np.array(filtered_df['Group category'].values.tolist())

####1C, Part A
x= np.array(filtered_df.iloc[:,[5,6,10,11,12,13]].values.tolist())
log_regression_1c(x, y)

###1C, Part B
x= np.array(filtered_df.iloc[:,[5,6,10,11,12]].values.tolist())
log_regression_1c(x, y)
x= np.array(filtered_df.iloc[:,[5,6,10,11,13]].values.tolist())
log_regression_1c(x,y)
x= np.array(filtered_df.iloc[:,[5,6,10,12,13]].values.tolist())
log_regression_1c(x,y)
x= np.array(filtered_df.iloc[:,[5,6,11,12,13]].values.tolist())
log_regression_1c(x,y)
x= np.array(filtered_df.iloc[:,[5,10,11,12,13]].values.tolist())
log_regression_1c(x,y)
x= np.array(filtered_df.iloc[:,[6,10,11,12,13]].values.tolist())
log_regression_1c(x,y)

###1C PART C, VALUE OF y has been changed
y = np.array(df['Group category'].values.tolist())
x= np.array(df.iloc[:,[5,6,10,11,12,13]].values.tolist())
log_regression_1c(x, y)

### Question 2
label = 'No-response conversation ratio'
exercise_2(df,label)
label = 'Night conversation ratio'
exercise_2(df,label)
label = 'Images ratio'
exercise_2(df,label)
    
   


