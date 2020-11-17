# Load in the data from JSON file
import pandas as pd

data = pd.read_json('input/restaurant.json')
data.head()

menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones", "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese", "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",
        "Prosciutto", "Salami"]

import spacy
from spacy.matcher import PhraseMatcher

index_of_review_to_test_on = 14
text_to_test_on = data.text.iloc[index_of_review_to_test_on]

# Load the SpaCy model
nlp = spacy.blank('en')

# Create the tokenized version of text_to_test_on
review_doc = nlp(text_to_test_on)

# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Create a list of tokens for each item in the menu
menu_tokens_list = [nlp(item) for item in menu]

matcher.add("MENU", menu_tokens_list)

# Find matches in the review_doc
matches = matcher(review_doc)

for match in matches:
    print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")

from collections import defaultdict

# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,
# the key is added with an empty list as the value.
item_ratings = defaultdict(list)

for idx, review in data.iterrows():
    doc = nlp(review['text'])
    # Using the matcher from the previous exercise
    doc_matches = matcher(doc)

    # Create a set of the items found in the review text
    found_items = {doc[match[1]:match[2]].lower_ for match in doc_matches}

    # Update item_ratings with rating for each item in found_items
    # Transform the item strings to lowercase to make it case insensitive
    for item in found_items:
        item_ratings[item].append(review['stars'])

# Calculate the mean ratings for each menu item as a dictionary
mean_ratings = {item: sum(ratings) / len(ratings) for (item, ratings) in item_ratings.items()}

# Find the worst item, and write it as a string in worst_item. This can be multiple lines of code if you want.
worst_item = sorted(mean_ratings, key=mean_ratings.get)[0]

print(mean_ratings, worst_item)
