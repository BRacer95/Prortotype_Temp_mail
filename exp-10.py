import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# Sample data structure as seen in notes
# Review, label
# "This movie was enjoyable", positive
# "I hated this movie", negative
# "Amazing movie", positive
# "worst movie ever", negative
# "really good film", positive
# "terrible acting", negative

# Load dataset
data = pd.read_csv("imdb_reviews.csv")

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['review'])
y = data['label']

# Train SVM model
model = SVC(kernel='linear')
model.fit(X, y)

# Predict new movie review
review = input("Enter a movie review: ")
# Transform the review text for prediction
review_vector = vectorizer.transform([review])
prediction = model.predict(review_vector)

if prediction[0] == "positive":
    print("positive Review")
else:
    print("negative Review")
