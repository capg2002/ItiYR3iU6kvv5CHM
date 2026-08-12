import numpy as np
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import re 
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

def clean_linkedin_location(location):
    if pd.isna(location):
        return None

    location = str(location).strip()

    # "Greater Boston Area" -> "Boston"
    location = re.sub(
        r"^Greater\s+",
        "",
        location,
        flags=re.IGNORECASE
    )

    # "Houston, Texas Area" -> "Houston, Texas"
    location = re.sub(
        r"\s+Area$",
        "",
        location,
        flags=re.IGNORECASE
    )

    return location

stop_words = set(stopwords.words('english'))
geolocator = Nominatim(user_agent="my_geo_standardizer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

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

talents_db["job_title"] = talents_db["job_title"].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
)

# Replace 500+ with 500

talents_db["connection"] = talents_db['connection'].replace("500+ ", 500)

talents_db["clean_location"] = talents_db["location"].apply(
    clean_linkedin_location
)

unique_locations = talents_db["clean_location"].dropna().unique()

location_mapping = {}

total = len(unique_locations)

for i, location in enumerate(unique_locations, start=1):

    print(f"[{i}/{total}] Geocoding: {location}")

    result = geocode(location)

    if result:
        location_mapping[location] = {
            "address": result.address,
            "latitude": result.latitude,
            "longitude": result.longitude
        }

        print(f"    -> {result.address}")
        print(f"    -> ({result.latitude}, {result.longitude})")

    else:
        location_mapping[location] = {
            "address": None,
            "latitude": None,
            "longitude": None
        }

        print("    -> No match")

talents_db["standard_address"] = talents_db["clean_location"].map(
    lambda x: location_mapping[x]["address"]
)

talents_db["latitude"] = talents_db["clean_location"].map(
    lambda x: location_mapping[x]["latitude"]
)

talents_db["longitude"] = talents_db["clean_location"].map(
    lambda x: location_mapping[x]["longitude"]
)


print(talents_db["job_title"])

talents_db.to_csv('filtered_talents_db.csv', index=False)