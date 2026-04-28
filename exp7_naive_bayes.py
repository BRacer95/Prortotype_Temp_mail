from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = ["win money now", "limited offer", "hello friend", "let's meet tomorrow"]
labels = ["spam", "spam", "ham", "ham"]

vec = CountVectorizer()
X = vec.fit_transform(emails)
model = MultinomialNB().fit(X, labels)

test = vec.transform(["congratulations you won"])
print(f"Naive Bayes Prediction: {model.predict(test)}")