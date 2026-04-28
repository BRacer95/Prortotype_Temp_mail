import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# Sample data structure as seen in notes
# text, label
# "I love this product", positive
# "this is amazing", positive
# "I hate this item", negative
# "worst experience", negative
# "very happy with service", positive
# "not good at all", negative

# Load dataset
data = pd.read_csv("sentiment.csv") 

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train SVM model
model = SVC(kernel='linear')
model.fit(X, y)

# Predict new sentence
sentence = input("Enter a sentence: ")
# Transform input to the same vector space
sentence_vector = vectorizer.transform([sentence])
prediction = model.predict(sentence_vector)

if prediction[0] == "positive":
    print("positive sentiment")
else:
    print("negative sentiment")
