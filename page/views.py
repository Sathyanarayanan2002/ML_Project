from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def home(request):
    return render(request,'home.html')

def ans(request):
    
    area1=request.POST['Squareft']
    bhk=request.POST['uiBHK']
    bath=request.POST['uiBathrooms']
    loc=request.POST['loc']
    dataframe=pd.read_csv("finaldoc.csv")
    X = dataframe.drop(['price'],axis='columns')
    Y=dataframe.price
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,y_train)
    loc_index = np.where(X.columns==loc)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = area1
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    price=lr_clf.predict([x])[0]
    Ans=round(price,2)
    return render(request,'ans.html',{'ans':Ans,'area':area1,'bath':bath,'loc':loc,'bhk':bhk})
