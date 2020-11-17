import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg')
review_data = pd.read_csv('input/yelp_ratings.csv')

review = """I absolutely love this place. The 360 degree glass windows with the 
Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
transports you to what feels like a different zen zone within the city. I know 
the price is slightly more compared to the normal American size, however the food 
is very wholesome, the tea selection is incredible and I know service can be hit 
or miss often but it was on point during our most recent visit. Definitely recommend!

I would especially recommend the butternut squash gyoza."""


def cosine_similarity(a, b):
    return np.dot(a, b) / np.sqrt(a.dot(a) * b.dot(b))


nlp = spacy.load('en_core_web_lg')

# Loading all document vectors from file
vectors = np.load('input/review_vectors.npy')

review_vec = nlp(review).vector

## Center the document vectors
# Calculate the mean for the document vectors, should have shape (300,)
vec_mean = vectors.mean(axis=0)
# Subtract the mean from the vectors
centered = vectors - vec_mean

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the review vector
sims = np.array([cosine_similarity(review_vec - vec_mean, centered_review) for centered_review in centered])

# Get the index for the most similar document
most_similar = sims.argmax()

print(review_data.iloc[most_similar].text)
