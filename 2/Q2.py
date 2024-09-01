from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize LDA and fit it to the training data
lda = LDA()
lda.fit(X_train, y_train)

# Predict the test set results
y_pred = lda.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate sensitivity and specificity
report = classification_report(y_test, y_pred, output_dict=True)

# Extracting sensitivity and specificity for each class
sensitivity = {}
specificity = {}

for label, metrics in report.items():
    if label.isdigit():  # Check if the key is a digit to ignore 'accuracy', 'macro avg', etc.
        sensitivity[label] = metrics['recall']
        
        # Specificity is the true negative rate
        true_negatives = cm.sum() - (cm[int(label), :].sum() + cm[:, int(label)].sum() - cm[int(label), int(label)])
        specificity[label] = true_negatives / (true_negatives + cm[:, int(label)].sum() - cm[int(label), int(label)])

# Print sensitivity and specificity for each class
print("\nSensitivity (Recall) for each class:")
for label, value in sensitivity.items():
    print(f"Class {label}: {value}")

print("\nSpecificity for each class:")
for label, value in specificity.items():
    print(f"Class {label}: {value}")
