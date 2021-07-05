Stacked Models
---

Attached contains code used to stack models in Python. Please note no hyperparameter tuning has been completed in this code!

***Base models:***  
Random Forest (default model)  
KNN (default model)  
XGBoost (default model)  
CatBoost (default model)  

***Meta learner model:***  
XGBoost (default model)

```python
# Define the base models
level0 = list()

level0.append(('knn', KNeighborsClassifier()))
level0.append(('r_forest', RandomForestClassifier( random_state=1)))
level0.append(('XGB', XGBClassifier(n_jobs=-1)))
level0.append(('CB', CatBoostClassifier()))

# define meta learner model
level1 = XGBClassifier()

# define the stacking ensemble - the stacked model is assessed using 10 cross validations
stk_mdl = StackingClassifier(estimators=level0, final_estimator=level1, cv=10, n_jobs=-1, verbose=2)
```

To extract the predicted probabilities one needs to use predict_proba

```python
classifier.predict_proba(X_test)[22]
# Original model prediction - 79% probability of Virginica and 21% probability of Versicolor

stk_mdl.predict_proba(X_test)[22]
# Stacked model prediction -  88% probability of Virginica and 11.5% probability of Versicolor
```
