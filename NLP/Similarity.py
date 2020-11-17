import numpy as np
import spacy


def cosine_similarity(a, b):
    return np.dot(a, b) / np.sqrt(a.dot(a) * b.dot(b))


nlp = spacy.load('en_core_web_lg')

a = nlp("REPLY NOW FOR FREE TEA").vector
b = nlp(
    "According to legend, Emperor Shen Nung discovered tea when leaves from a wild tree blew into his pot of boiling water.").vector
print(cosine_similarity(a, b))
