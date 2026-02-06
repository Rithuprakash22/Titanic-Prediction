import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.write("App Started")
st.title("🚢 Titanic Survival Prediction App")

df = pd.read_csv(r"C:\Users\rithu\Desktop\TITANIC\Titanic-Dataset.csv")

st.write("Reached Before Model Training")

X = df[['Pclass','Sex','Age','SibSp','Embarked']].copy()
X.loc[:, 'Sex'] = X['Sex'].map({'male':1,'female':0})
X.loc[:, 'Embarked'] = X['Embarked'].map({'C':0,'Q':1,'S':2})
X = X.fillna(0)

y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=55
)

log = LogisticRegression(max_iter=1000)
log.fit(x_train,y_train)

st.write("Model Trained Successfully")

# Sidebar Inputs
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class",[1,2,3])
sex = st.sidebar.selectbox("Gender",["Male","Female"])
age = st.sidebar.slider("Age",2,80,25)
family = st.sidebar.number_input("Family Members",0,10,1)
embarked = st.sidebar.selectbox("Embarked",["S","C","Q"])

sex = 1 if sex=="Male" else 0
embarked_map = {"C":0,"Q":1,"S":2}
embarked = embarked_map[embarked]

input_data = np.array([[pclass,sex,age,family,embarked]])

if st.button("Predict Survival"):
    prediction = log.predict(input_data)

    if prediction[0]==1:
        st.success("Passenger Survived ✅")
    else:
        st.error("Passenger Did Not Survive ❌")
