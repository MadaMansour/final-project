# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target labels (Setosa, Versicolour, Virginica)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN model
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(iris.target_names)), iris.target_names, rotation=45)
plt.yticks(np.arange(len(iris.target_names)), iris.target_names)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

# Optional: Create a Streamlit interface (if required)
# Save this script as a .py file and run it using: streamlit run script_name.py
# Uncomment the lines below to use Streamlit

# import streamlit as st
# st.title('Iris Flower Classification with KNN')
# sepal_length = st.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()))
# sepal_width = st.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()))
# petal_length = st.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()))
# petal_width = st.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()))
# input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# prediction = knn.predict(input_data)
# st.write('Predicted Class:', iris.target_names[prediction[0]])
