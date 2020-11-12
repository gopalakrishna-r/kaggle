import pandas as pd

spam = pd.read_csv("input/spam.csv")

# Building bag of words model
import spacy
nlp = spacy.blank("en") # empty model

#Create the textcategorizer with exclusive classes and bow architecture
textcat = nlp.create_pipe("textcat",
                          config={
                              "exclusive_classes":True,
                              "architecture":"bow"
                          })

nlp.add_pipe(textcat)

# add label
textcat.add_label("ham")
textcat.add_label("spam")

# create dictionary of labels
train_texts = spam['text'].values
train_labels = [{'cats':{'ham':label == 'ham',
                         'spam':label == 'spam'}}
                         for label in spam['label']]

train_data = list(zip(train_texts, train_labels))
print(train_data[:3])

# train the model
from spacy.util import minibatch
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

import random
losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # create a batch generator
    batches = minibatch(train_data, 8)
    for batch in batches:
        texts, labels = zip(*batch)
        nlp.update(texts,labels,sgd = optimizer, losses = losses)
    print(losses)

# make predictions
texts = ["Care for lunch?" ]
docs = [nlp.tokenizer(text) for text in texts]

textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

print(scores)

predicted_labels = scores.argmax(axis = 1)
print([textcat.labels[label] for label in predicted_labels])