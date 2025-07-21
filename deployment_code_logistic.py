import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Sex = st.sidebar.selectbox('Sex',('1','0'))
    Pclass = st.sidebar.selectbox("Passanger Class",('1','2','3'))
    Embarked = st.sidebar.selectbox("Embarked: S=1,C=2,Q=3", ('0','1','2'))
    Age = st.sidebar.number_input("Insert the Age")
    SibSp = st.sidebar.number_input("Insert Siblings Sp")
    Fare = st.sidebar.number_input("Insert Fare")
    Parch = st.sidebar.number_input("Insert Parch")
    data = {'Sex':Sex,
            'Embarked':Embarked,
            'Pclass':Pclass,
            'Age':Age,
            'SibSp':SibSp,
            'Parch':Parch,
            'Fare':Fare}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

Survived = pd.read_csv("Titanic_train.csv")
Survived.drop(["PassengerId","Name","Ticket","Cabin"],inplace=True,axis = 1)
for col in Survived.columns:
    if Survived[col].dtype != 'O':
        median_value = Survived[col].median()
        Survived[col] = Survived[col].fillna(median_value)
Survived = Survived.dropna()

SS= StandardScaler()
LE = LabelEncoder()
SS_X = SS.fit_transform(Survived[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
SS_X= pd.DataFrame(SS_X)
SS_X.columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

LE_X_sex= LE.fit_transform(Survived[['Sex']])
LE_X_sex = pd.DataFrame(LE_X_sex)
LE_X_sex.columns=['Sex']

LE_X_Embarked= LE.fit_transform(Survived[['Embarked']])
LE_X_Embarked = pd.DataFrame(LE_X_Embarked)
LE_X_Embarked.columns=['Embarked']

X = pd.concat([LE_X_sex,LE_X_Embarked,SS_X], axis=1)
Y = Survived.iloc[:,0]

clf = LogisticRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')


st.subheader('Prediction Probability')
st.write(prediction_proba)
