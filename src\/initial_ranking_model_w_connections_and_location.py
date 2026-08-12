from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Loading cleaned data from eda.py
data = pd.read_csv("filtered_talents_db.csv")

# Treat id as string (for future functionality)
data["id"] = data["id"].astype(str)

# Define ranking history columns and dataframe
rank_col_names = ['Change Number', 'ID(s)', 'Change Type', 'Old Score', 'Updated Score',
                  'Old Rank', 'New Rank']
ranking_history = pd.DataFrame(columns=rank_col_names)
k = 10

# Define ranking algorithm performance metrics dataframe
feedback_metric_columns = [
    "Change Number",
    "Good Feedback Count",
    "Bad Feedback Count",
    "Mean Good Model Rank",
    "Mean Bad Model Rank",
    f"Good Recall@{k}",
    "Good-Bad Pairwise Accuracy",
    "Good-Bad Score Margin"
]

feedback_metrics_history = pd.DataFrame(
    columns=feedback_metric_columns
)

# Load a pre-trained CrossEncoder model (SSTB Roberta-base was chosen as a normalized 0-1 scale encoder)
ce_model = CrossEncoder("cross-encoder/stsb-roberta-base")

# Initializing global state for reranking tracking
change_counter = 0 
good_inputs = []
bad_inputs = []

geolocator = Nominatim(user_agent="my_geo_standardizer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def normalize_for_column(text):
    ## Used to normalize column names to distinguish between inputs.

    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")[:30]
        # Cut off at 30 characters to avoid overly long names (they're long enough as is)
    )

def update_ranking_history(changelog, rank_table, updated_input, change_type, old_score, new_score,
                           old_rank, new_rank):
    ## Updates the ranking history dataframe with new changes to the ranking queries

    # Counter for total number of user based update requests
    number_of_updates = len(updated_input)

    # If this is the initial call, do not update ranking history
    if number_of_updates == 0:
        return(rank_table)

    # Add new data. np.asarray ensures order is preserved
    new_data = pd.DataFrame({'Change Number': [changelog] * number_of_updates,
                             'ID(s)': list(updated_input), 
                             'Change Type': [change_type] * number_of_updates,
                             'Old Score': np.asarray(old_score), 
                             'Updated Score': np.asarray(new_score),
                             'Old Rank': np.asarray(old_rank),
                             'New Rank': np.asarray(new_rank)})

    # Concatenate new data dataframe with rank table
    rank_table = pd.concat([rank_table, new_data], ignore_index=True)

    # Returns updated ranking tracking table
    return(rank_table)

def ask_location():
    while True:
        job_location = input("Where is the work located? ").strip()

        final_job_loc = geocode(job_location)

        if not final_job_loc:
            print("Location could not be found. Please try again.")
            continue

        print(
            f"The parsed address is {final_job_loc.address} "
            f"with coordinates "
            f"({final_job_loc.latitude}, {final_job_loc.longitude})"
        )

        job_loc_dec = input("Is this correct? Type Y if so. ")

        if job_loc_dec.strip().casefold() in ["y", "yes"]:
            print(
                "Location was selected. Profiles closer to "
                "the location will be favored."
            )

            return (
                final_job_loc.address,
                final_job_loc.latitude,
                final_job_loc.longitude
            )


            
def new_query():

    # Initialize local variables for initial query for the baseline of the algorithm
    hr_query = []
    last_input = str()

    # Ensures input is valid before moving on to the next step.
    while last_input != ['END']:
        entry = input("Enter queries separated by commas, "
                "'hr' for defaults, or 'END' to finish: ")

        # Converts input into legible Python list
        last_input = [item.strip() for item in entry.split(',')]

        # User can see what was interpreted by the programme
        print(last_input)

        # If user types hr, get the default sample use given by the problem
        if last_input == ['hr']:
            hr_query = ["Aspiring human resources", "seeking human resources"]

            # Breaks while loop
            break

        # If user does not type END, it keeps adding more user inputs
        elif last_input != ['END']:
            hr_query += last_input

        # If they write END, programme stops taking new baseline queries
        else: 
            break
    return(hr_query)

def make_final_scores(db, desc):
    ## Create aggregate columns to create final scores for each algorithmic query

    # Rename columns to be well-defined
    score_columns = [
        column
        for column in db.columns
        if column.startswith(f"score_{desc}_")
    ]

    # Failsafe if db doesn't have any columns starting with the given name
    if not score_columns:
        print(f"No {desc} columns found")
        return db

    # Max and mean are preserved in the database to form as aggregate scores each round
    db[f"final_{desc}_score_mean"] = db[score_columns].mean(axis=1)
    db[f"final_{desc}_score_max"] = db[score_columns].max(axis=1)

    return(db)

def fit_tfidf_to_query(query_input, title_col, db, desc):
    ## TF-IDF fitting for the query, judging pairwise cosine similarity between expected inputs

    queries = [str(query).strip() for query in query_input if str(query).strip()]

    # Failsafe if there are no queries in query_input after stripping it
    if not queries:
        print("No queries were found")
        return(db)

    # Candidate titles with failsafes for future datasets, in case there are missig values
    candidate_titles = (db[title_col]
                        .fillna("").astype(str).str.strip().replace("", "unknown"))

    # Total vector of all titles with the base queries
    combined_corpus = (candidate_titles.tolist() + queries)

    # Maintains all as lowercase using unigrams and bigrams. Normalizes using classic dot prod.
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1,2), sublinear_tf=True, norm="l2"
    )

    # Fitting TF-IDF on total vector. Vectorizes sentence based on similarity to uni and bigrams
    tfidf_matrix = vectorizer.fit_transform(combined_corpus)

    # Candidate matrix is the uni- and bigram vectorizations for the candidates
    candidate_matrix = tfidf_matrix[:len(db)]

    # Candidate matrix is the uni- and bigram vectorizations for the search queries
    query_matrix = tfidf_matrix[len(db):]

    # Calculates cosine similarity between candidates and queries
    similarities = cosine_similarity(candidate_matrix, query_matrix)

    # Creates TF-IDF columns for the database
    for query_index, query in enumerate(queries):
        column_name = (f"score_{desc}_"f"{normalize_for_column(query)}")

        db[column_name] = similarities[:, query_index]

    db = make_final_scores(db, desc)

    # Returns db with final TF-IDF similarity to queries
    return(db)

def fit_model_to_query(query_input, title_col, model, db, desc):
    ## Predict scores for a pair of sentences between queries and candidates

    
    for query in query_input:
        print("THIS IS CURRENT QUERY:", query)

        # Creates list pairs of every query and their candidate, then fits
        # cross encoder for each
        pairs = [[query, text] for text in db[title_col]]
        scores = model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )

        similarity_scores = np.clip(
            np.asarray(scores, dtype=float),
            0.0,
            1.0
        )

        column_name = (
            f"score_{desc}_{normalize_for_column(query)}"
        )

        db[column_name] = similarity_scores

    if len(query_input) > 0:


        db = make_final_scores(db, desc)

    # Returns db with the final scores for cross encoder sentence similarity
    return(db)

def add_distance_from_preferred_loc(db, loc_cols, loc_coords):

    lat_col, lon_col = loc_cols

    db["distance_from_loc"] = db.apply(lambda row: geodesic(loc_coords, (row[lat_col], row[lon_col])).km,
                                       axis = 1)

    minimum = db["distance_from_loc"].min()
    maximum = db["distance_from_loc"].max()

    db["location_score"] = 1 - (
        (db["distance_from_loc"] - minimum)
        / (maximum - minimum)
    )

    return(db)

def combine_similarity_scores(db, ce_desc, tfidf_desc, output_desc, dist_col, ce_weight = 0.65, tfidf_weight = 0.3, loc_weight = 0.05):

    # Define total model weight to check it is always valid
    total_weight = ce_weight + tfidf_weight + loc_weight

    if total_weight <= 0:
        raise ValueError("Similarity weights must sum to a positive value.")

    for statistic in ["mean", "max"]:

        # Creates FINAL mean and max columns for cross encoder, TF-IDF, and them combined with weights
        ce_column = (f"final_{ce_desc}_score_{statistic}")
        tfidf_column = (f"final_{tfidf_desc}_score_{statistic}")
        combined_column = (f"final_{output_desc}_score_{statistic}")
        db[combined_column] = (loc_weight * db[dist_col] + ce_weight * db[ce_column] + tfidf_weight * db[tfidf_column]) / total_weight

    # Outputs database with combined model scores
    return(db)

def add_fit_inputs(db, id_col, fit_type):
    ## Allows user to add new "good" and "bad" fits to positively and negatively influence the model's
    # bias towards that naming

    decision_input = input(
        f"Would you like to add {fit_type} fit indices? Type Y if yes. "
    )

    if decision_input.strip().casefold() in ["y", "yes"]:
        new_inputs = input(
            "Input a list of indices separated by commas. "
        )

        last_input = [
            item.strip()
            for item in new_inputs.split(",")
        ]

        valid_ids = set(db[id_col])

        # Ensures user inputs a list of existing indices in the dataset
        while not set(last_input).issubset(valid_ids):
            print(last_input)
            new_inputs = input("Please input a valid list of indices separated by commas. ")

            last_input = [item.strip() for item in new_inputs.split(",")]

        # Confirms input was successful
        print(f"{fit_type.capitalize()} inputs successfully inputted!")

        return last_input

    return []

def update_feedback_metrics(metrics_table, db, id_col, score_col, good_ids, bad_ids, change_number, k=10):
    ## Updates feedback metric table

    # Ranks score column
    ranked = (
        db.sort_values(
            score_col,
            ascending=False
        )
        .reset_index(drop=True)
        .copy()
    )

    ranked["evaluation_rank"] = (
        np.arange(len(ranked)) + 1
    )

    # Indexes the ranked list
    indexed = ranked.set_index(id_col)

    # Verifies current good queries
    available_good = [
        candidate_id
        for candidate_id in good_ids
        if candidate_id in indexed.index
    ]

    # Verifies current bad queries
    available_bad = [
        candidate_id
        for candidate_id in bad_ids
        if candidate_id in indexed.index
    ]

    # Extracts good scores only
    good_scores = indexed.loc[available_good, score_col].to_numpy(dtype=float)

    # Extracts bad scores only
    bad_scores = indexed.loc[available_bad,score_col].to_numpy(dtype=float)

    # Extracts good evaluation ranks only
    good_ranks = indexed.loc[available_good,"evaluation_rank"].to_numpy(dtype=float)

    # Extracts bad evaluation ranks only
    bad_ranks = indexed.loc[available_bad,"evaluation_rank"].to_numpy(dtype=float)

    # Extracts the top k (10 by default) IDs to check if the good ranked queries are there
    top_k_ids = set(
        ranked.head(k)[id_col]
    )

    # Check the proportion of good ids that are in the top k, then calculate the mean good rank
    if available_good:
        good_recall_at_k = (
            len(top_k_ids.intersection(available_good))
            / len(available_good)
        )
        mean_good_rank = good_ranks.mean()

    # If fails, return NA
    else:
        good_recall_at_k = np.nan
        mean_good_rank = np.nan

    # Find mean rank for bad ids
    if available_bad:
        mean_bad_rank = bad_ranks.mean()
    else:
        mean_bad_rank = np.nan

    # If there have been bad and good queries, figure out proportion of good queries greater than bad queries in rank
    if available_good and available_bad:
        greater_than = (good_scores[:, None]> bad_scores[None, :])

        equal_to = (good_scores[:, None] == bad_scores[None, :])

        pairwise_accuracy = np.mean(greater_than + 0.5 * equal_to)

        score_margin = (good_scores.mean() - bad_scores.mean())
    else:
        pairwise_accuracy = np.nan
        score_margin = np.nan

    # Creating new row for feedback table
    new_row = pd.DataFrame([{
        "Change Number": change_number,
        "Good Feedback Count": len(available_good),
        "Bad Feedback Count": len(available_bad),
        "Mean Good Model Rank": mean_good_rank,
        "Mean Bad Model Rank": mean_bad_rank,
        f"Good Recall@{k}": good_recall_at_k,
        "Good-Bad Pairwise Accuracy": (pairwise_accuracy),
        "Good-Bad Score Margin": score_margin
    }])

    # Returns concatenated updated feedback table
    return pd.concat([metrics_table, new_row], ignore_index=True)

def normalize_match(value):
    ## Used to normalize user input for reranking algo
    return str(value).strip().casefold()[:30]

def reranking_algo(db, id_col, title_col, base_col, ce_model):
    ## Algorithm to rerank all candidates given new good and bad queries

    global good_inputs
    global bad_inputs
    global change_counter
    global ranking_history
    global feedback_metrics_history

    dec_loc = input("Would you like to change the location?")
    if dec_loc.strip().casefold() in ["y", "yes"]:
        new_location = ask_location()

        new_coords = (
            new_location[1],
            new_location[2]
        )

        db = add_distance_from_preferred_loc(
            db,
            ["latitude", "longitude"],
            new_coords
        )

        print("Location has been updated.")
    else:
        print("Location has remained the same.")

    # Saves previous scores column based on whether rerankings have been attempted previously or not
    previous_score_column = ("final_fit_with_weights_mean"
        if "final_fit_with_weights_mean" in db.columns
        else base_col)

    # Saves previous scores using conditional column name and sets ids as the actual index
    previous_scores = (db
                       .set_index(id_col)[previous_score_column]
                       .copy()
                       )

    # Saves previous ranks using conditional column name and sets ids as the actual index
    previous_ranks = (db
                      .set_index(id_col)["rank"]
                      .copy()
                      )

    # Identifies previously created columns and erases them to introduce the new scores
    feedback_columns = [column for column in db.columns
        if (column.startswith((
                "score_ce_good_",
                "score_ce_bad_",
                "score_tfidf_good_",
                "score_tfidf_bad_",
                "final_ce_good_",
                "final_ce_bad_",
                "final_tfidf_good_",
                "final_tfidf_bad_",
                "final_good_",
                "final_bad_",
                "final_fit_"
            ))
            or column in [
                "model_fit_score",
                "model_rank"
            ]
        )
    ]

    # Drops aforementioned columns
    db = db.drop(
        columns=feedback_columns,
        errors="ignore"
    )

    # User-facing option to remove conditions
    remove_inputs = input("Would you like to remove any conditions? Type Y if so.")
    if remove_inputs.strip().casefold() in ["y","yes"]:

        # Ensures users can remove both titles and ids, depending on their column access
        removed_inputs = input("Input target titles or IDs you want to remove conditions for, " \
        "separated by commas.")

        # Translates input into list
        removed_inputs = [normalize_match(item) for item in removed_inputs.split(',') if item.strip()]

        # Valid inputs are in the id column or the title column
        valid_inputs = ({normalize_match(value) for value in db[id_col]} | {normalize_match(value) for value in db[title_col]}
)
        # Ensures valid inputs are input
        while not set(removed_inputs).issubset(valid_inputs):
            print(removed_inputs)
            removed_inputs = input(
                "Please input a valid list of indices separated by commas. "
            )

            removed_inputs = [normalize_match(item) for item in removed_inputs.split(",")]

        # Initialize matching columns for removed ids/titles depending on what user gives
        matching_cols = []

        titles = db.loc[db[id_col].isin(removed_inputs), title_col]
        removed_inputs += titles.tolist()

        # Matching columns are columns that have a name containing the normalized column, so it can be removed
        for col in removed_inputs:
            normalized_col = normalize_for_column(col)

            matching_cols += db.filter(
                like=normalized_col
            ).columns.tolist()

        removed_ids = [item for item in removed_inputs if item in set(db[id_col])]

        # Keeps track of removed ids and adds them to a dictionary
        removed_ids += db.loc[db[title_col].isin(removed_inputs), id_col].tolist()

        removed_ids = list(dict.fromkeys(removed_ids))

        # Creates id list for good and bad inputs from global variables, removing them from the lists
        good_inputs = [candidate_id for candidate_id in good_inputs if candidate_id not in removed_ids]

        bad_inputs = [candidate_id for candidate_id in bad_inputs if candidate_id not in removed_ids]

        # Drops the removed columns
        db = db.drop(columns=matching_cols, errors="ignore")

    # Requests user for good inputs
    good_input = add_fit_inputs(db, id_col, "good")

    # Requests user for bad inputs
    bad_input = add_fit_inputs(db, id_col, "bad")

    # Updates global lists tracking good and bad ids
    good_inputs.extend(good_input)
    bad_inputs.extend(bad_input)

    good_inputs = list(dict.fromkeys(good_inputs))
    bad_inputs = list(dict.fromkeys(bad_inputs))

    # Ensures they are mutually exclusive
    good_inputs = [candidate_id for candidate_id in good_inputs if candidate_id not in bad_input]
    bad_inputs = [candidate_id for candidate_id in bad_inputs if candidate_id not in good_input]

    # Saves old scores and ranks of good and bad inputs
    old_scores_g = (previous_scores
                    .reindex(good_input)
                    .to_numpy()
                    )

    old_scores_b = (previous_scores
                    .reindex(bad_input)
                    .to_numpy()
                    )

    old_scores_rank_g = (previous_ranks
                        .reindex(good_input)
                        .to_numpy()
                        )

    old_scores_rank_b = (previous_ranks
                        .reindex(bad_input)
                        .to_numpy()
                        )

    # Saves the titles of good and bad ids

    good_titles = (db.loc[db[id_col].isin(good_inputs), title_col]
                   .dropna()
                   .astype(str)
                   .drop_duplicates()
                   .tolist()
                   )

    bad_titles = (db.loc[db[id_col].isin(bad_inputs), title_col]
                 .dropna()
                 .astype(str)
                 .drop_duplicates()
                 .tolist()
                 )

    # Initialize a temporary database before committing to the main db
    db_temp = db

    # If good titles were chosen, fit cross encoder, TF-IDF and combine scores
    if good_titles:
        db_temp = fit_model_to_query(
            good_titles,
            title_col,
            ce_model,
            db_temp,
            "ce_good"
        )

        db_temp = fit_tfidf_to_query(
            good_titles,
            title_col,
            db_temp,
            "tfidf_good"
        )

        db_temp = combine_similarity_scores(
            db_temp,
            ce_desc="ce_good",
            tfidf_desc="tfidf_good",
            output_desc="good",
            dist_col= "location_score",
            ce_weight=0.70,
            tfidf_weight=0.30
        )

    # If bad titles were chosen, fit cross encoder, TF-IDF and combine scores
    if bad_titles:
        db_temp = fit_model_to_query(
            bad_titles,
            title_col,
            ce_model,
            db_temp,
            "ce_bad"
        )

        db_temp = fit_tfidf_to_query(
            bad_titles,
            title_col,
            db_temp,
            "tfidf_bad"
        )

        db_temp = combine_similarity_scores(
            db_temp,
            ce_desc="ce_bad",
            tfidf_desc="tfidf_bad",
            output_desc="bad",
            dist_col= "location_score",
            ce_weight=0.70,
            tfidf_weight=0.30
        )

    # Calculate avoidance to penalize candidates that are LIKE the bad queries
    if "final_bad_score_max" in db_temp.columns:
        bad_avoidance = (1 - db_temp["final_bad_score_max"])

    weighted_score = (0.45 * db_temp[base_col])

    active_weight = 0.45

    # Incorporate number of connections as a small part of influence. Should be reviewed by HR.

    db_temp["norm_connection"] = (db_temp["connection"]/500)
    weighted_score += (0.05 * db_temp["norm_connection"])
    active_weight += 0.05

    # Adds weighted scores depending on what is/is not included in the user-entered requirements
    if "final_good_score_max" in db_temp.columns:
        weighted_score += (0.35 * db_temp["final_good_score_max"])
        active_weight += 0.35

    if "final_bad_score_max" in db_temp.columns:
        bad_avoidance = (1 - db_temp["final_bad_score_max"])

        weighted_score += (0.15 * bad_avoidance)
        active_weight += 0.15

    # Sets a proper model fit score using weighted score with the active weight to normalize the results
    db_temp["model_fit_score"] = (weighted_score / active_weight)

    # Obtains new rank using the model fit
    db_temp["model_rank"] = db_temp["model_fit_score"].rank(ascending=False,method="min")

    indexed_temp = db_temp.set_index(id_col)

    # Obtains new scores and ranks following the model fit
    new_scores_g = (indexed_temp
                    .reindex(good_input)["model_fit_score"]
                    .to_numpy()
                    )

    new_scores_b = (indexed_temp
                    .reindex(bad_input)["model_fit_score"]
                    .to_numpy()
                    )

    new_scores_rank_g = (indexed_temp
                         .reindex(good_input)["model_rank"]
                         .to_numpy()
                         )

    new_scores_rank_b = (indexed_temp
                         .reindex(bad_input)["model_rank"]
                         .to_numpy()
                         )

    # Ensures the final fit with weights is the same as model fit score
    db_temp["final_fit_with_weights_mean"] = db_temp["model_fit_score"].copy()
    
    # Update change counter
    change_counter += 1

    # Update feedback table
    feedback_metrics_history = (
        update_feedback_metrics(
            metrics_table=feedback_metrics_history,
            db=db_temp,
            id_col=id_col,
            score_col="model_fit_score",
            good_ids=good_inputs,
            bad_ids=bad_inputs,
            change_number=change_counter,
            k=k
        )
    )

    # Sets good inputs to a score of 1, as requested and as they were actively accepted by HR
    db_temp.loc[db_temp[id_col].isin(good_inputs), "final_fit_with_weights_mean"] = 1.0

    # Sets bad inputs to a score of 0 as actively rejected by HR
    db_temp.loc[db_temp[id_col].isin(bad_inputs),"final_fit_with_weights_mean"] = 0.0

    # Creates rank for the final fit weights to keep naming consistent
    db_temp["rank"] = db_temp["final_fit_with_weights_mean"].rank(ascending=False, method="min")

    db = db_temp.sort_values(["rank"], ascending=[True])
    
    if good_input:

        # Updates ranking table for good inputs
        ranking_history = update_ranking_history(change_counter, 
                                                ranking_history,
                                                good_input,
                                                "Good",
                                                old_scores_g,
                                                new_scores_g,
                                                old_scores_rank_g,
                                                new_scores_rank_g)
    if bad_input:
        # Updates ranking table for bad inputs

        ranking_history = update_ranking_history(change_counter, 
                                                    ranking_history,
                                                    bad_input,
                                                    "Bad",
                                                    old_scores_b,
                                                    new_scores_b,
                                                    old_scores_rank_b,
                                                    new_scores_rank_b)

    # Returns reranked master database
    return(db)


def add_initial_relevance_gate(db,ce_column=("final_ce_base_score_mean"),
                               tfidf_column=("final_tfidf_base_score_max"),
                               semantic_floor_quantile=0.30):
    # Creates a baseline relevance gate, as per request such that, if the score is lower than the semantic floor, 
    # it does not pass. These values are not removed outright as HR should review these.
    semantic_floor = db[ce_column].quantile(semantic_floor_quantile)

    # Passes if cross encoding is higher than the set floor or its TF-IDF is positive, meaning it shares 
    # some cosine similarity (not leaning dissimilar)
    db["passes_initial_gate"] = ((db[ce_column] >= semantic_floor) | (db[tfidf_column] > 0))

    return db, semantic_floor

# User requested for query
my_query = new_query()

position_location = []

# Get position location
position_location = ask_location()
position_coords = (position_location[1], position_location[2])

data = add_distance_from_preferred_loc(data, ["latitude", "longitude"], position_coords)

# Cross Encoding base scores
data = fit_model_to_query(my_query,
                          "job_title",
                          ce_model,
                          data,
                          "ce_base")

# TF-IDF base scores
data = fit_tfidf_to_query(my_query, "job_title", data, "tfidf_base")

# Combined base scores. CE is given higher weight as it relies on a pre-trained big data
# sentence transformation model that has been trained on word relation, which is
# stronger than the similarity index vectorization proposed by TF-IDF. 
# 0.7 and 0.3 were chosen as a strong enough relative weight of importance.

data = combine_similarity_scores(data,
                                 ce_desc="ce_base",
                                 tfidf_desc="tfidf_base",
                                 output_desc="base",
                                 dist_col = "location_score",
                                 ce_weight=0.70,
                                 tfidf_weight=0.30)


# Defines and sorts by base rank
data["rank"] = data["final_base_score_mean"].rank(ascending=False,method="min")

data = data.sort_values(["rank"],ascending=True)

# Establishes the relevance gate, adding the column to the data
data, initial_semantic_floor = (add_initial_relevance_gate(data))

# Run reranking script
data = reranking_algo(data, "id", "job_title", "final_base_score_mean", ce_model)

# Marker to end rerank or not
end_reranking = 0

while end_reranking == 0:

    # Asks user if they want to continue reranking, then reruns reranking script if so
    want_rerank = input("Do you want to continue adding changes? Type Y if so.")

    if want_rerank.strip().casefold() in ["y", "yes"]:
        data = reranking_algo(data, "id", "job_title", "final_base_score_mean", ce_model)

    else:
        end_reranking = 1

# Saves updated master db to CSV
data.to_csv('trial_ranking.csv', index=False)

# Saves current ranking history to CSV
ranking_history.to_csv("ranking_history.csv", index=False)

print(ranking_history)

# Saves feedback table to CSV
feedback_metrics_history.to_csv("feedback_metrics_history.csv", index=False)

initially_excluded = data.loc[
    ~data["passes_initial_gate"],
    [
        "id",
        "job_title",
        "final_ce_base_score_mean",
        "final_tfidf_base_score_max",
        "final_base_score_mean"
    ]
]

only_value_and_rank = data.loc[:, ["id","job_title", "rank"]]

only_value_and_rank.to_csv("only_value_and_rank.csv")

# Prints all initially excluded entries
print(initially_excluded)