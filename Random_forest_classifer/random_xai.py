from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_diabetes
import pandas as pd

diabetes = pd.read_csv('diabetes.csv')

X = diabetes.drop(columns = 'Outcome')
y = diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

rf_clf = RandomForestClassifier(max_features=2, n_estimators = 100, bootstrap = True)

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

print(classification_report(y_test, y_pred))

import shap 
shap.initjs()

import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(rf_clf)

shap_val = explainer.shap_values(X_test)

fig = plt.figure()
shap.summary_plot(shap_val, X_test)