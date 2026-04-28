from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

reviews = ["I love this", "This is bad", "I hate it", "Great service"]
labels = ["pos", "neg", "neg", "pos"]

vec = CountVectorizer()
X = vec.fit_transform(reviews)
model = SVC(kernel='linear').fit(X, labels)

test = vec.transform(["I am happy"])
print(f"SVM Sentiment: {model.predict(test)}")