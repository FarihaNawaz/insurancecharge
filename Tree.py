# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:08:33 2018

@author: Prithila
"""
from sklearn import tree 
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
#reading the data
data = pd.read_csv('insurance.csv')
new_charges=data['charges']
new_children=data['children']
new_age=data['age']
new_bmi=data['bmi']
size=len(new_charges)
#finding the mean to convert data 
x=data['charges'].mean()
y=data['children'].mean()
z=data['age'].mean()
a=data['bmi'].mean()
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
data['charges'] = new1
data['children'] = new2
data['age'] = new3
data['bmi']=new4
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
predictors = ['age','sex','bmi','children','smoker']
#take the predictions
X = data[predictors] 
Y = data.charges 
#training the dataset
X_train, X_test , y_train,y_test = train_test_split(X, Y, test_size = 0.1, random_state = 5000) 
#making the decission tree
decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy") 
decisionTreeClassifier.fit(X_train.astype(int),y_train.astype(int)) 
y_pred = decisionTreeClassifier.predict(X_test)
# Making the confusion matrix 
cm = confusion_matrix(y_test,y_pred)
TP=cm[0,0]#true positive
FN=cm[0,1]#false negative
FP=cm[1,0]#false positive
TN=cm[1,1]#true negative
total=sum(sum(cm))
#from confusion matrix calculate accuracy
accuracy1=(TP+TN)/total
print ('Accuracy : ', accuracy1*100,"%")
sensitivity1 = TP/(TP+FN) #hit rate
print('Sensitivity : ', sensitivity1*100 ,"%")
specificity1 =TN/(TN+FP)
print('Specificity : ', specificity1*100,"%")
PPV = TP/(TP+FP)
print('Positive Predicted value : ', PPV)
NPV = TN/(TN+FN)
print('Negative Predicted Value : ', NPV)
FNR = FN/(FN+TP)                                            
print('False Negative Rate : ', FNR)##miss rate
FPR=FP/(FP+TN)
print('False positive Rate: ',FPR)#fall-out
#roc curve
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#dTree = decisionTreeClassifier.fit(X, Y) 
#print(dTree)
#dotData = tree.export_graphviz(dTree, out_file=None) 
#print(dotData)