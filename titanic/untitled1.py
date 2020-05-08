import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn



dataset=pd.read_csv('train.csv')
dataset.head()


dataset['Family_Size']=dataset['Parch']+dataset['SibSp']+1
sorted(dataset['Family_Size'].unique())


def process_family():
    global dataset
        
    #introducing features based on family size
    dataset['Singleton']=dataset['Family_Size'].map(lambda s:1 if s==1 else 0)
    dataset['Small_Family']=dataset['Family_Size'].map(lambda s:1 if  s>1 and s<=5 else 0)
    dataset['Large_Family']=dataset['Family_Size'].map(lambda s:1 if s>5 else 0)
    dataset.drop(['Parch','SibSp'],axis=1,inplace=True)


process_family()
dataset.head()


def process_embark():
    global dataset
    dataset.Embarked.fillna('S',inplace=True)
    pd_dummies=pd.get_dummies(dataset['Embarked'],prefix='Embarked')
    dataset=pd.concat([dataset,pd_dummies],axis=1)
    dataset.drop('Embarked',axis=1,inplace=True)
    return dataset

process_embark()


def process_cabin():
    global dataset
    dataset.Cabin.fillna('T',inplace=True)
    dataset['Cabin']=dataset['Cabin'].map(lambda c:c[0])
    pd_dummies=pd.get_dummies(dataset['Cabin'],prefix='Cabin')
    dataset=pd.concat([dataset,pd_dummies],axis=1)
    dataset.drop('Cabin',axis=1,inplace=True)
    return dataset

process_cabin()



Title_Dictionary={
    "Capt":"Officer",
    "Col":"Officer",
    "Don":"Royalty",
    "Dr":"Officer",
    "Jonkheer":"Royalty",
    "Lady":"Royalty",
    "Major":"Officer",
    "Master":"Master",
    'Miss':"Miss",
    'Mlle':"Miss",
    'Mme':"Mrs",
    'Mr':"Mr",
    'Mrs':"Mrs",
    'Ms':"Miss",
    'Rev':"Officer",
    'Sir':"Master",
    'the Countess':"Royalty"
}



def get_titles():
    dataset["Title"]=dataset['Name'].map(lambda a:a.split(',')[1].split('.')[0].strip())
    
    dataset['Title']=dataset.Title.map(Title_Dictionary)
    
    return dataset


dataset=get_titles()
dataset.head()

a=dataset.Age.mean()
dataset['Age'].fillna(a,inplace=True)


dataset.drop('Name',axis=1,inplace=True)
title_dummy=pd.get_dummies(dataset['Title'],prefix='Title')
dataset=pd.concat([dataset,title_dummy],axis=1)
dataset.drop('Title',axis=1,inplace=True)


#dataset.drop(['Died'],axis=1,inplace=True)

title_dummy=pd.get_dummies(dataset['Sex'],prefix='Sex')
dataset=pd.concat([dataset,title_dummy],axis=1)
dataset.drop('Sex',axis=1,inplace=True)

y=dataset.Survived.to_numpy()
dataset.drop(['Survived','Ticket'],axis=1,inplace=True)
X=dataset




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)




from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=180,min_samples_leaf=3,max_features=0.5,n_jobs=-1)
classifier.fit(X_train,y_train)
classifier.score(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
cr=classification_report(y_pred,y_test)

cm=confusion_matrix(y_pred,y_test)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm1=confusion_matrix(y_pred,y_test)






