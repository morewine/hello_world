# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:56:58 2018

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

""" 自定义函数：对数据图像进行打印输出 """
def show(x,y):
    plt.scatter(x[y==0,0],x[y==0,1])
    plt.scatter(x[y==1,0],x[y==1,1])
    plt.scatter(x[y==2,0],x[y==2,1])
    plt.show()

""" 为了方便，我们导入自带的鸢尾花的数据集，且只取后两个维度的数据 """
iris = datasets.load_iris()
x = iris.data[:,2:]
y = iris.target
show(x,y)

""" 训练决策树的分类器，决策树的划分标准使用基尼不纯度 (INI impurity) """ 
dt_clf = DecisionTreeClassifier(max_depth =3 ,criterion='gini')
dt_clf.fit(x,y)

""" 自定义函数：绘测决策边界 """
def plot_decision_boundary(model, axis):
    x0,x1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*500)).reshape((-1,1)), #这里的reshape()中的参数可能要改
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*500)).reshape((-1,1))
    )
    x_new = np.c_[x0.ravel(),x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0,x1,zz,linewidth=5,cmap=custom_cmap)     

plot_decision_boundary(dt_clf,axis=[0.5,7.5,0,3])
show(x,y)
                       
