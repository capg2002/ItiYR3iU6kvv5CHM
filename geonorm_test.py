import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import re

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


geolocator = Nominatim(user_agent="my_geo_standardizer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


data = pd.read_csv("potential-talents - Aspiring human resources - seeking human resources.csv")

data["clean_location"] = data["location"].apply(
    clean_linkedin_location
)

unique_locations = data["clean_location"].dropna().unique()

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

data["standard_address"] = data["clean_location"].map(
    lambda x: location_mapping[x]["address"]
)

data["latitude"] = data["clean_location"].map(
    lambda x: location_mapping[x]["latitude"]
)

data["longitude"] = data["clean_location"].map(
    lambda x: location_mapping[x]["longitude"]
)

data.to_csv("geonorm_test_output.csv")
