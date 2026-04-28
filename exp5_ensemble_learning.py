from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('rf', RandomForestClassifier())], voting='hard')
model.fit(X_train, y_train)
print(f"Ensemble Prediction for [0.5, 0.5]: {model.predict([[0.5, 0.5]])}")