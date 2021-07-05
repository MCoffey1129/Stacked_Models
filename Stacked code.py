"""Below sets out a template I use for stacking models."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

"""# Import the iris dataset"""
iris = sns.load_dataset("iris")

"""Typical queries used to evaluate your data - always carry this out before completing any analysis
    on your data"""
iris.head()
iris.info()
iris.describe()
iris.columns
iris.isnull().sum() # there are no null values in the data

"""Check the correlation between each of the vars"""
"""Sepal length and Sepal width as well as petal length and petal with look to be highly correlated"""
sns.pairplot(iris)
iris.corr()

################################################################################################################
                # Random Forest used to predict the species
################################################################################################################

"""# Importing the dataset"""
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

"""# Splitting the dataset into the Training set and Test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

"""# Feature Scaling"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

"""# Training the Random Forest model on the Training data (without tuning the hyperparameters)"""
classifier = RandomForestClassifier(random_state = 1)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only one case which was misclassified"""
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy_score(y_test, y_pred)


################################################################################################################
                # Stacking the models
################################################################################################################


# Define the base models
level0 = list()

level0.append(('knn', KNeighborsClassifier()))
level0.append(('r_forest', RandomForestClassifier( random_state=1)))
level0.append(('XGB', XGBClassifier(n_jobs=-1)))
level0.append(('CB', CatBoostClassifier()))

# define meta learner model
level1 = XGBClassifier()

# define the stacking ensemble
stk_mdl = StackingClassifier(estimators=level0, final_estimator=level1, cv=10, n_jobs=-1, verbose=2)

# fit the model on all available data
stk_mdl.fit(X_train, y_train)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only one case which is still misclassified"""
y_stk_pred = stk_mdl.predict(X_test)
cm = confusion_matrix(y_test, y_stk_pred)
print(cm)
print(classification_report(y_test, y_stk_pred))


################################################################################################################
                # Investigate the probabilities of the misclassified case
                # for both the original model and the stacked model
################################################################################################################

orig_model_pred =pd.concat([pd.DataFrame(y_pred,columns=['Predicted']) , pd.DataFrame(y_test,columns=['Actual'])]
                           , axis=1)
stk_model_pred =pd.concat([pd.DataFrame(y_stk_pred,columns=['Predicted']) , pd.DataFrame(y_test,columns=['Actual'])]
                           , axis=1)

orig_model_pred.loc[orig_model_pred['Predicted'] != orig_model_pred['Actual']]
                            # row 22 predicted Virginica actually versicolor

stk_model_pred.loc[stk_model_pred['Predicted'] != stk_model_pred['Actual']]
                            # row 22 predicted Virginica actually versicolor


classifier.predict_proba(X_test)[22]
# Original model prediction - 79% probability of Virginica and 21% probability of Versicolor

stk_mdl.predict_proba(X_test)[22]
#Stacked model prediction -  88% probability of Virginica and 11.5% probability of Versicolor

"""You can see from the graph below that the misclassified case looks like a Virginica
   and we would not expect the model the predict otherwise"""
# plt.clf()
sns.set()
_=sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue='species')
_=plt.annotate("Misclassified case", (5.0, 1.7))
plt.plot()

