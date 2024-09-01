from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit multinomial logistic regression model

clf_with_intercept = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)
clf_without_intercept = LogisticRegression(multi_class='multinomial', fit_intercept=False, solver='lbfgs').fit(X_train, y_train)

# Predict on test set
pred_with_intercept = clf_with_intercept.predict(X_test)
pred_without_intercept = clf_without_intercept.predict(X_test)

# Compute confusion matrices
conf_mat_with_intercept = confusion_matrix(y_test, pred_with_intercept)
conf_mat_without_intercept = confusion_matrix(y_test, pred_without_intercept)

# Print results
print("With intercept:\n")
print(conf_mat_with_intercept)
print(classification_report(y_test, pred_with_intercept))

print("\nWithout intercept:\n")
print(conf_mat_without_intercept)
print(classification_report(y_test, pred_without_intercept))