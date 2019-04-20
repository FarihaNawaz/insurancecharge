# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:27:14 2018

@author: Prithila
"""
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc
#reading the data
data = pd.read_csv('insurance.csv')
df = pd.DataFrame(data,columns=['age','sex','bmi','children','smoker','region','charges'])
df=df.drop(['region'],axis=1)
new_charges=df['charges']
new_children=df['children']
new_age=df['age']
new_bmi=df['bmi']
size=len(new_charges)
#calculation the median to convert data
x=df['charges'].median()
print(x)
y=df['children'].median()
z=df['age'].median()
a=df['bmi'].median()
new1=[]
new2=[]
new3=[]
new4=[]
#binarize the data
for i in range(0,size):
    if (new_bmi[i]>a):
        new4.append(1) 
    else:
        new4.append(0)
for i in range(0,size):
    if (new_age[i]>z):
        new3.append(1) 
    else:
        new3.append(0)
for i in range(0,size):
    if (new_children[i]>y):
        new2.append(1) 
    else:
        new2.append(0)
for i in range(0,size):
    if (new_charges[i]>x):
        new1.append(1) 
    else:
        new1.append(0)
df['charges'] = new1
df['children'] = new2
df['age'] = new3
df['bmi']=new4
df['sex'] = data['sex'].map({'male': 0, 'female': 1})
df['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
X=df[['age','sex','bmi','children','smoker']]
Y=df['charges']
#split the training and testing
X_train, X_test , y_train,y_test = train_test_split(X, Y, test_size = 0.1, random_state = 500)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#using logistic regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#creating the confusion matrix
cm = confusion_matrix(y_test,y_pred)
TP=cm[0,0]#true positive
FN=cm[0,1]#false negative
FP=cm[1,0]#false positive
TN=cm[1,1]#true negative
total=sum(sum(cm))
#from confusion matrix calculating the accuracy
accuracy1=(TP+TN)/total
print ('Accuracy : ', accuracy1*100,"%")
sensitivity1 = TP/(TP+FN)
print('Sensitivity : ', sensitivity1*100 ,"%")
specificity1 =TN/(TN+FP)
print('Specificity : ', specificity1*100,"%")
PPV = TP/(TP+FP)
print('Positive Predicted value : ', PPV)
NPV = TN/(TN+FN)
print('Negative Predicted Value : ', NPV)
FNR = FN/(FN+TP)                                            
print('False Negative Rate : ', FNR)
FPR=FP/(FP+TN)
print('False positive Rate: ',FPR)
#roc curve
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)