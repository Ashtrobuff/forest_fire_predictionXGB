import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import*
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pickle

#loading our forest fire pre-processed data
data = pd.read_csv('fflag.csv')

#printing columns(optional)
print(data.columns)

# Separate the target variable from the features
X = data.drop(['IncidentOccurred','ForestFiresDate'],axis=1)
y = data['IncidentOccurred']

#Label encode the target variable (only for non string values but we have processed it accordingly)
#le = LabelEncoder()
#y = le.fit_transform(y)

#splitting the forestfire data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=null)

#importing the model 
model = xgb.XGBClassifier(  max_depth=6,learning_rate=0.1,n_estimators=100,)
modeldt = DecisionTreeClassifier()

#Train the model
model.fit(X_train, y_train)

#modeldt.fit(X_train,y_train)

with open('my_model2.pkl', 'wb') as f:
    pickle.dump(model, f)


with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Creating a numpy array to predict the target value(trial)
data=np.array([633,3,3,88,2022,0,0,0,0,0,0,0,0,20600,10051,0,0,0,0,0,0])
predictions = model.predict(np.array([data]))
#print the predictions
print(predictions)

#------Accuracy Analysis------#
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
acc_diff = abs(train_acc - test_acc)
print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)
print("Accuracy difference:", acc_diff)

# Check if the model is overfitting
if acc_diff > 0.05:
    print("The model is overfitting.")
else:
    print("The model is not overfitting.")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

# Evaluating performance using ROC curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print('ROC AUC:', roc_auc)

# Evaluating performance using confusion matrix(optional)
cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix:')
print(cm)
xgb.plot_importance(model, importance_type='gain')

# Show plot
plt.show()

