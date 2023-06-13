import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#===================================INPUT======================================
#import dataset
Dataset=pd.read_csv('Kdd_dataset.csv')
Dataset.shape
print (Dataset.head())

#================================PRE_PROCESSING================================
#Data Preprocessing
#check missing values
print ('Dataset contain null:\t',Dataset.isnull().values.any())
print ('Describe null:\n',Dataset.isnull().sum())
print ('No of  null:\t',Dataset.isnull().sum().sum())


#Selecting Independent variable 
x=Dataset.drop("class",axis = 1).values
x1=pd.DataFrame(x)

#Selecting Dependent variable
y=Dataset['class'].values
k1=pd.DataFrame(y)

#class lable converting
for i in range(9999):
    if y[i]=='normal':
        y[i]=0
    else:
        y[i]=1
type(y)
#type(x)
y=y.astype('int')


#==============================LABEL ENCOADING=================================
#Encoding categorical data

labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
Y=pd.DataFrame(x[:,1])

labelencoder_x=LabelEncoder()
x[:,2]=labelencoder_x.fit_transform(x[:,2])
Y=pd.DataFrame(x[:,2])

labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
Y=pd.DataFrame(x[:,3])
ct = ColumnTransformer([("protocol_type", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)
labelencoder_x=LabelEncoder()
ct = ColumnTransformer([("service", OneHotEncoder(), [2])], remainder = 'passthrough')
x = ct.fit_transform(x)
ct = ColumnTransformer([("flag", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
#=============================MODEL SELECTION==================================
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.75,random_state=0)


'''RANDOM FOREST'''

#Create a Gaussian Classifier
rf_clf=RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train,y_train)
rf_ypred=rf_clf.predict(X_test)
print('\n')
print("------Accuracy------")
rf=accuracy_score(y_test, rf_ypred)*100
RF=('RANDOM FOREST Accuracy:',accuracy_score(y_test, rf_ypred)*100,'%')
print(RF)
print('\n')
print("------Classification Report------")
print(classification_report(rf_ypred,y_test))
print('\n')
print('Confusion_matrix')
rf_cm = confusion_matrix(y_test, rf_ypred)
print(rf_cm)
print('\n')
tn = rf_cm[0][0]
fp = rf_cm[0][1]
fn = rf_cm[1][0]
tp = rf_cm[1][1]
Total_TP_FP=rf_cm[0][0]+rf_cm[0][1]
Total_FN_TN=rf_cm[1][0]+rf_cm[1][1]
specificity = tn / (tn+fp)
rf_specificity=format(specificity,'.3f')
sensitivity = tp / (fn + tp)
rf_sensitivity=format(sensitivity,'.3f')
print('RF_specificity:',rf_specificity)
print('\n')
print('RF_sensitivity:',rf_sensitivity)
print('\n')
plt.figure()
skplt.estimators.plot_learning_curve(RandomForestClassifier(n_estimators=100), X_train, y_train,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Random Forest Digits Classification Learning Curve");

plt.figure()                                   
sns.heatmap(confusion_matrix(y_test,rf_ypred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


'''Decision Tree'''

print('Decision Tree')
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)      
dt_pred=dt.predict(X_test)
DT=accuracy_score(dt_pred,y_test)*100
print("Decision Tree Accuracy is: ", DT,'%')
dt_cm=confusion_matrix(dt_pred,y_test)
print("------Classification Report------")
print(classification_report(rf_ypred,y_test))
print('\n')
print('DT Confusion Matrix')
print(dt_cm)
tn = dt_cm[0][0]
fp = dt_cm[0][1]
fn = dt_cm[1][0]
tp = dt_cm[1][1]
Total_TP_FP=dt_cm[0][0]+dt_cm[0][1]
Total_FN_TN=dt_cm[1][0]+dt_cm[1][1]
specificity = tn / (tn+fp)
dt_specificity=format(specificity,'.3f')
sensitivity = tp / (fn+tp)
dt_sensitivity=format(sensitivity,'.3f')
print('DT_specificity:',dt_specificity)
print('\n')
print('DT_sensitivity:',dt_sensitivity)

skplt.estimators.plot_learning_curve(DecisionTreeClassifier() , X_train, y_train,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Decision Tree Digits Classification Learning Curve");

plt.figure()                                  
sns.heatmap(confusion_matrix(y_test,dt_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

'''support vector classifier'''

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
model=SVC(C=1000,gamma=0.0001,kernel='rbf')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('support vector classifier')
sv=SVC(gamma=0.001)
print("-----Accuracy Score-----")
print("support vector machine = ",accuracy_score(y_test,y_pred)*100,'%')
print('\n')
print("------Classification Report------")
print(classification_report(y_pred,y_test))
print('\n')
print('Confusion_matrix')
sv_cm = confusion_matrix(y_test, y_pred)
print(sv_cm)
print('\n')
tn = sv_cm[0][0]
fp = sv_cm[0][1]
fn = sv_cm[1][0]
tp = sv_cm[1][1]
Total_TP_FP=sv_cm[0][0]+sv_cm[0][1]
Total_FN_TN=sv_cm[1][0]+sv_cm[1][1]
specificity = tn / (tn+fp)
sv_specificity=format(specificity,'.3f')
sensitivity = tp / (fn + tp)
sv_sensitivity=format(sensitivity,'.3f')
print('sv_specificity:',sv_specificity)
print('\n')
print('sv_sensitivity:',sv_sensitivity)
print('\n')
plt.figure()
skplt.estimators.plot_learning_curve(SVC(gamma=0.001), X_train, y_train,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Support vector Classification Learning Curve");
plt.figure()
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


prediction = rf_clf.predict([[0,1,1,1,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,16,1,0,1,1,0,0,0,0,0,1,18]])
print(prediction)







