import numpy as np
import pandas as pd

import re 
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

talents_db = pd.read_csv("potential-talents - Aspiring human resources - seeking human resources.csv")
# Note that all job titles must be standardized. There will be clear NLP done to 
# ensure comparisons are consistent. 

char_frequencies = Counter("".join(talents_db["job_title"].str.lower()))

print(char_frequencies)

# Note that there are several symbols aside from letters. There are spaces, 
# full stops, commas, parentheses, numbers, dashes, exclamation marks, and 
# new line commands. All of these should be removed insofar as it does not 
# remove context (like with numbers saying "HR Rep 1" and the like).

number_rows = talents_db["job_title"].str.contains(r'\d')

print(talents_db.loc[number_rows, "job_title"])

# Note that, for the most part, they come from all the duplicate copies of
# 2019 C.T. Bauer College of Business Graduate (Magna Cum Laude) and aspiring Human Resources professional,
# and the numbers do not impact the meaning of the entry.

# On row 75, it includes a phone number. 
# On row 99, it includes a graduation year.
# All of these cases are irrelevant to their positions. Thus, all punctuation can be 
# removed.

talents_db = talents_db.drop_duplicates(subset=['job_title', 'location', 'connection'])

talents_db["job_title"] = talents_db["job_title"].str.replace(r'[^a-zA-Z ]', '', regex=True).str.lower()

unigram_counts = talents_db["job_title"]

vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=None)

ngram_matrix = vectorizer.fit_transform(talents_db['job_title'].dropna())

frequencies = ngram_matrix.sum(axis=0).A1
ngram_names = vectorizer.get_feature_names_out()

df_counts = pd.DataFrame({'N-gram': ngram_names, 'Count': frequencies})
df_counts = df_counts.sort_values(by='Count', ascending=False).reset_index(drop=True)

print(df_counts[df_counts['Count'] > 10])
# Note all cases of "aspiring" comes from peopple claiming "aspiring human resources"
# and every cases of "human" and "resources" comes from "human resources"

print(talents_db["job_title"])

talents_db.to_csv('filtered_talents_db.csv', index=False)