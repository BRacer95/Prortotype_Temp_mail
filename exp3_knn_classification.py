import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    'Age': [35, 30, 33, 40, 45, 20, 25, 62, 48, 33, 20],
    'Salary': [160000, 60000, 55000, 70000, 80000, 40000, 62000, 97000, 90000, 20000, 52000],
    'Purchased': [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0]
})

X, y = data[['Age', 'Salary']], data['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print(f"KNN Accuracy: {accuracy_score(y_test, model.predict(X_test))}")