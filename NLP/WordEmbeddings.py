import numpy as np
import spacy

nlp = spacy.load("en_core_web_lg")

text = "These vectors can be used as features for machine learning models."
with nlp.disable_pipes():
    vectors = np.array([token.vector for token in nlp(text)])

print(vectors.shape)

import pandas as pd

spam = pd.read_csv('input/spam.csv')

with nlp.disable_pipes():
    doc_vectors = np.array(([nlp(text).vector for text in spam.text]))

print(doc_vectors.shape)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(doc_vectors, spam.label, test_size=0.1, random_state=1)

from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy {svc.score(X_valid, y_valid) * 100:.3f}%")
