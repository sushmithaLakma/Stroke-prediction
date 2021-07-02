import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title('Stroke prediction')
st.write('"This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status."')

data = pd.read_csv('stroke-data.csv')

st.write('#### Shape of dataset :',data.shape)
st.text("")
st.text("")

if st.checkbox('Columns & Datatypes'):
	col1_1, col1_2 = st.beta_columns(2)
	for column in data.columns:
		col1_1.write(column)
	for dtype in data.dtypes:	
		col1_2.write(dtype)		

st.text("")
if st.checkbox('Show few rows of dataset'):
	st.write(data.head())	

st.text("")
if st.checkbox('Look at the distribution of the data'):
	st.write(data.describe())	

option = st.sidebar.selectbox('Visualisation of data',['gender','ever_married','work_type','Residence_type','smoking_status'])
if option == 'gender':
	st.write(px.histogram(data, x="gender", color="stroke"))
elif option == 'ever_married':
	st.write(px.histogram(data, x="ever_married", color="stroke"))
elif option == 'work_type':
	st.write(px.histogram(data, x="work_type", color="stroke"))
elif option == 'Residence_type':
	st.write(px.histogram(data, x="Residence_type", color="stroke"))
elif option == 'smoking_status':		
	st.write(px.histogram(data, x="smoking_status", color="stroke"))

# Oversampling
os = RandomOverSampler(sampling_strategy=0.5)

X = data.drop('stroke',axis=1)
y = data['stroke']

X_res,y_res = os.fit_resample(X,y)

col2_1, col2_2 = st.beta_columns(2)

with col2_1:
	st.write('### It\'s an imbalanced dataset.') 
	st.write('Original dataset shape', Counter(y))
	fig = px.histogram(y, x='stroke',opacity=0.4,width=400,height=500)
	st.write(fig.update_layout(bargap=0.2))

with col2_2:	
	st.write('### After oversampling : ')
	st.write('Resampled dataset shape', Counter(y_res))
	fig = px.histogram(y_res, x='stroke',opacity=0.7,width=400,height=500)
	st.write(fig.update_layout(bargap=0.2))

X_res.drop(['id','Residence_type'],axis=1,inplace=True)

X_res = pd.get_dummies(X_res,drop_first=True)

# Imputing with mean values for bmi
mean = X_res['bmi'].mean()
X_res['bmi'] = X_res['bmi'].fillna(mean)

#X_res['work_type_Self_employed'] = X_res['work_type_Self-employed']

X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state = 42)

# Check with other classifiers
#st.text("")
#st.sidebar.write('Accuracy scores of Classifiers')
#if st.sidebar.checkbox('Logistic Regression'):
#	st.write('> Accuracy score using Logistic Regression',round(accuracy_score(y_test, prediction) * 100, 2),'%')	
#elif st.sidebar.checkbox('Decision Tree Classifier'):
#	st.write('> Accuracy score using Decision Tree Classifier',round(accuracy_score(y_test, prediction) * 100, 2),'%')
#elif st.sidebar.checkbox('Random Forest Classifier'):
#	st.write('> Accuracy score using Random Forest Classifier',round(accuracy_score(y_test, prediction) * 100, 2),'%')
#elif st.sidebar.checkbox('Pipeline model with Random Forest Classifier'):
#	st.write('> Accuracy score using Pipeline model with Random Forest Classifier',round(accuracy_score(y_test, prediction) * 100, 2),'%')

#Pipeline with StandardScaler and RFC
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

# Predictions
st.write('### > Check if you\'re likely to get a stroke!')
st.text("")
gender = st.radio('Gender',['Male','Female','Other'])
st.text("")
age = st.number_input('Age (in years)',1,100)
st.text("")
hypertension = st.radio('Hypertension (BP)',[0,1])
st.text("")
heart_disease = st.radio('Any heart related complications',[0,1])
st.text("")
ever_married = st.selectbox('Married or not ?',options=['Yes','No'])
st.text("")
work_type = st.selectbox('Work type ?',options=['Private','Self-employed','Govt_job','children','Never_worked'])
st.text("")
avg_glucose_level = st.slider('Glucose level',0,300)
st.text("")
bmi = st.slider('BMI',10,100)
st.text("")
smoking_status = st.selectbox('Do you smoke ?',options=['Formerly smoked','Never smoked','Smokes','Unknown'])

# Gender
if gender == 'Male':
	gender_Male = 1
	gender_Other = 0
elif gender == 'Other':
	gender_Male = 0
	gender_Other = 1
else:
	gender_Male = 0
	gender_Other = 0	

# Ever married
if ever_married == 'Yes':
	ever_married_Yes = 1
else:
	ever_married_Yes = 0

# Work type	
if work_type == 'Private':
	work_type_Private = 1
	work_type_Never_worked = 0
	work_type_Self_employed = 0
	work_type_children = 0
elif work_type == 'Self-employed':
	work_type_Private = 0
	work_type_Never_worked = 0
	work_type_Self_employed = 1
	work_type_children = 0
elif work_type == 'Govt_job':
	work_type_Private = 0
	work_type_Never_worked = 0
	work_type_Self_employed = 0
	work_type_children = 0
elif work_type == 'Children':
	work_type_Private = 0
	work_type_Never_worked = 0
	work_type_Self_employed = 0
	work_type_children = 1
else:
	work_type_Private = 0
	work_type_Never_worked = 1
	work_type_Self_employed = 0
	work_type_children = 0		

# Smoking status
if smoking_status == 'Formerly smoked':
	smoking_status_formerly_smoked = 1
	smoking_status_never_smoked = 0
	smoking_status_smokes = 0
elif smoking_status == 'Never smoked':
	smoking_status_formerly_smoked = 0
	smoking_status_never_smoked = 1
	smoking_status_smokes = 0
elif smoking_status == 'Smokes':
	smoking_status_formerly_smoked = 0
	smoking_status_never_smoked = 0
	smoking_status_smokes = 1
else:
	smoking_status_formerly_smoked = 0
	smoking_status_never_smoked = 0
	smoking_status_smokes = 0		

predict_stroke = pipeline.predict([[age,hypertension,heart_disease,avg_glucose_level,bmi,
	gender_Male,gender_Other,ever_married_Yes,work_type_Never_worked,
	work_type_Private,work_type_Self_employed,work_type_children,
	smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes]])

st.text("")
if st.button('Chance of a stroke?'):
	st.write(predict_stroke[0])

st.text("")
metric_option = st.sidebar.radio('Metrics with Random Forest Classifier',['Accuracy score','Confusion matrix','Classification report'])
if metric_option == 'Accuracy score':
	st.write('#### > Accuracy score :')
	st.write("")
	st.write(round(accuracy_score(y_test, prediction) * 100, 2),'%')
if metric_option == 'Confusion matrix':
	st.write('#### > Confusion matrix :')
	st.text("")
	st.write(confusion_matrix(y_test, prediction))	
if metric_option == 'Classification report':
	st.write('#### > Classification report :')
	st.text("")
	st.write(classification_report(y_test, prediction))	