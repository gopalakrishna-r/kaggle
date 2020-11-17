import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg')

review_data = pd.read_csv('input/yelp_ratings.csv')

reviews = review_data[:100]
print(reviews)
# with nlp.disable_pipes():
#     vectors = np.array([nlp(review_doc.text).vector for _,review_doc in reviews.iterrows()])
#
# print(vectors.shape)

# Loading all document vectors from file
vectors = np.load('input/review_vectors.npy')

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, random_state=1, test_size=0.1)

model = LinearSVC(random_state=1, dual=False)
model.fit(X_train, y_train)
print(f'Model test accuracy: {model.score(X_test, y_test) * 100:.3f}%')

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, random_state=1, test_size=0.1)

model = RandomForestClassifier(n_estimators=500,
                               bootstrap=False,
                               oob_score=False,
                               n_jobs=-1,
                               random_state=42)
model.fit(X_train, y_train)
print(f'Model test accuracy: {model.score(X_test, y_test) * 100:.3f}%')
