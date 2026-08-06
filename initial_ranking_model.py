from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("filtered_talents_db.csv")
data["id"] = data["id"].astype(str)

rank_col_names = ['Change Number', 'ID(s)', 'Change Type', 'Old Score', 'Updated Score',
                  'Old Rank', 'New Rank']
ranking_history = pd.DataFrame(columns=rank_col_names)

feedback_metric_columns = [
    "Change Number",
    "Good Feedback Count",
    "Bad Feedback Count",
    "Mean Good Model Rank",
    "Mean Bad Model Rank",
    "Good Recall@10",
    "Good-Bad Pairwise Accuracy",
    "Good-Bad Score Margin"
]

feedback_metrics_history = pd.DataFrame(
    columns=feedback_metric_columns
)

# Load a pre-trained CrossEncoder model
ce_model = CrossEncoder("cross-encoder/stsb-roberta-base")

change_counter = 0 
good_inputs = []
bad_inputs = []

def normalize_for_column(text):
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")[:30]
    )

def update_ranking_history(changelog, rank_table, updated_input, change_type, old_score, new_score,
                           old_rank, new_rank):
    number_of_updates = len(updated_input)

    if number_of_updates == 0:
        return(rank_table)
    
    new_data = pd.DataFrame({'Change Number': [changelog] * number_of_updates,
                             'ID(s)': list(updated_input), 
                             'Change Type': [change_type] * number_of_updates,
                             'Old Score': np.asarray(old_score), 
                             'Updated Score': np.asarray(new_score),
                             'Old Rank': np.asarray(old_rank),
                             'New Rank': np.asarray(new_rank)})
    rank_table = pd.concat([rank_table, new_data], ignore_index=True)

    return(rank_table)

def new_query():

    hr_query = []
    last_input = str()
    while last_input != ['END']:
        # hr_query = ["Aspiring human resources", "seeking human resources"]

        ## Add list inputs too!
        entry = input("Enter queries separated by commas, "
                "'hr' for defaults, or 'END' to finish: ")
        last_input = [item.strip() for item in entry.split(',')]
        print(last_input)
        if last_input == ['hr']:
            hr_query = ["Aspiring human resources", "seeking human resources"]
            break
        elif last_input != ['END']:
            hr_query += last_input
        else: 
            break
    return(hr_query)

def make_final_scores(db, desc):

    score_columns = [
        column
        for column in db.columns
        if column.startswith(f"score_{desc}_")
    ]

    if not score_columns:
        return db

    db[f"final_{desc}_score_mean"] = db[score_columns].mean(axis=1)
    db[f"final_{desc}_score_sum"] = db[score_columns].sum(axis=1)
    db[f"final_{desc}_score_max"] = db[score_columns].max(axis=1)
    db[f"final_{desc}_score_min"] = db[score_columns].min(axis=1)

    return(db)

def fit_tfidf_to_query(query_input, title_col, db, desc):
    queries = [str(query).strip() for query in query_input if str(query).strip()]

    if not queries:
        return db

    candidate_titles = (db[title_col]
                        .fillna("").astype(str).str.strip().replace("", "unknown"))

    combined_corpus = (candidate_titles.tolist() + queries)

    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1,2), sublinear_tf=True, norm="l2"
    )

    tfidf_matrix = vectorizer.fit_transform(combined_corpus)

    candidate_matrix = tfidf_matrix[:len(db)]
    query_matrix = tfidf_matrix[len(db):]

    similarities = cosine_similarity(candidate_matrix, query_matrix)

    for query_index, query in enumerate(queries):
        column_name = (f"score_{desc}_"f"{normalize_for_column(query)}")

        db[column_name] = similarities[:, query_index]

    db = make_final_scores(db, desc)

    return db

def fit_model_to_query(query_input, title_col, model, db, desc):
    # Predict scores for a pair of sentences
    for query in query_input:
        print("THIS IS CURRENT QUERY:", query)
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

    return(db)

def combine_similarity_scores(db, ce_desc, tfidf_desc, output_desc, ce_weight = 0.7, tfidf_weight = 0.3):
    total_weight = ce_weight + tfidf_weight

    if total_weight <= 0:
        raise ValueError("Similarity weights must sum to a positive value.")

    for statistic in ["mean", "max"]:

        ce_column = (f"final_{ce_desc}_score_{statistic}")
        tfidf_column = (f"final_{tfidf_desc}_score_{statistic}")
        combined_column = (f"final_{output_desc}_score_{statistic}")
        db[combined_column] = (ce_weight * db[ce_column] + tfidf_weight * db[tfidf_column]) / total_weight

    return(db)

def add_fit_inputs(db, id_col, fit_type):
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

        while not set(last_input).issubset(valid_ids):
            print(last_input)
            new_inputs = input(
                "Please input a valid list of indices separated by commas. "
            )

            last_input = [
                item.strip()
                for item in new_inputs.split(",")
            ]

        print(
            f"{fit_type.capitalize()} inputs successfully inputted!"
        )

        return last_input

    return []

def update_feedback_metrics(
    metrics_table,
    db,
    id_col,
    score_col,
    good_ids,
    bad_ids,
    change_number,
    k=10
):
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

    indexed = ranked.set_index(id_col)

    available_good = [
        candidate_id
        for candidate_id in good_ids
        if candidate_id in indexed.index
    ]

    available_bad = [
        candidate_id
        for candidate_id in bad_ids
        if candidate_id in indexed.index
    ]

    good_scores = indexed.loc[
        available_good,
        score_col
    ].to_numpy(dtype=float)

    bad_scores = indexed.loc[
        available_bad,
        score_col
    ].to_numpy(dtype=float)

    good_ranks = indexed.loc[
        available_good,
        "evaluation_rank"
    ].to_numpy(dtype=float)

    bad_ranks = indexed.loc[
        available_bad,
        "evaluation_rank"
    ].to_numpy(dtype=float)

    top_k_ids = set(
        ranked.head(k)[id_col]
    )

    if available_good:
        good_recall_at_k = (
            len(
                top_k_ids.intersection(
                    available_good
                )
            )
            / len(available_good)
        )

        mean_good_rank = good_ranks.mean()
    else:
        good_recall_at_k = np.nan
        mean_good_rank = np.nan

    if available_bad:
        mean_bad_rank = bad_ranks.mean()
    else:
        mean_bad_rank = np.nan

    if available_good and available_bad:
        greater_than = (
            good_scores[:, None]
            > bad_scores[None, :]
        )

        equal_to = (
            good_scores[:, None]
            == bad_scores[None, :]
        )

        pairwise_accuracy = np.mean(
            greater_than
            + 0.5 * equal_to
        )

        score_margin = (
            good_scores.mean()
            - bad_scores.mean()
        )
    else:
        pairwise_accuracy = np.nan
        score_margin = np.nan

    new_row = pd.DataFrame([{
        "Change Number": change_number,
        "Good Feedback Count": len(
            available_good
        ),
        "Bad Feedback Count": len(
            available_bad
        ),
        "Mean Good Model Rank": mean_good_rank,
        "Mean Bad Model Rank": mean_bad_rank,
        "Good Recall@10": good_recall_at_k,
        "Good-Bad Pairwise Accuracy": (
            pairwise_accuracy
        ),
        "Good-Bad Score Margin": score_margin
    }])

    return pd.concat(
        [metrics_table, new_row],
        ignore_index=True
    )

def reranking_algo(db, id_col, title_col, base_col, ce_model):
    global good_inputs
    global bad_inputs
    global change_counter
    global ranking_history
    global feedback_metrics_history
    

    previous_score_column = (
        "final_fit_with_weights_mean"
        if "final_fit_with_weights_mean" in db.columns
        else base_col
    )

    previous_scores = (
        db.set_index(id_col)[previous_score_column]
        .copy()
    )

    previous_ranks = (
        db.set_index(id_col)["rank"]
        .copy()
    )

    feedback_columns = [
        column
        for column in db.columns
        if (
            column.startswith((
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

    db = db.drop(
        columns=feedback_columns,
        errors="ignore"
    )

    remove_inputs = input("Would you like to remove any conditions? Type Y if so.")
    if remove_inputs.strip().casefold() in ["y","yes"]:
        removed_inputs = input("Input target titles or IDs you want to remove conditions for, " \
        "separated by commas.")

        removed_inputs = [item.strip() for item in removed_inputs.split(',')]

        valid_inputs = set(db[id_col]) | set(db[title_col])

        while not set(removed_inputs).issubset(valid_inputs):
            print(removed_inputs)
            removed_inputs = input(
                "Please input a valid list of indices separated by commas. "
            )

            removed_inputs = [
                item.strip()
                for item in removed_inputs.split(",")
            ]

        matching_cols = []

        titles = db.loc[db[id_col].isin(removed_inputs), title_col]
        removed_inputs += titles.tolist()
        for col in removed_inputs:
            normalized_col = normalize_for_column(col)

            matching_cols += db.filter(
                like=normalized_col
            ).columns.tolist()

        removed_ids = [
            item
            for item in removed_inputs
            if item in set(db[id_col])
        ]

        removed_ids += db.loc[
            db[title_col].isin(removed_inputs),
            id_col
        ].tolist()

        removed_ids = list(dict.fromkeys(removed_ids))

        good_inputs = [
            candidate_id
            for candidate_id in good_inputs
            if candidate_id not in removed_ids
        ]

        bad_inputs = [
            candidate_id
            for candidate_id in bad_inputs
            if candidate_id not in removed_ids
        ]
                
        db = db.drop(columns=matching_cols, errors="ignore")

    good_input = add_fit_inputs(db, id_col, "good")

    bad_input = add_fit_inputs(db, id_col, "bad")

    good_inputs.extend(good_input)
    bad_inputs.extend(bad_input)

    good_inputs = list(dict.fromkeys(good_inputs))
    bad_inputs = list(dict.fromkeys(bad_inputs))

    bad_inputs = [
        candidate_id
        for candidate_id in bad_inputs
        if candidate_id not in good_input
    ]

    good_inputs = [
        candidate_id
        for candidate_id in good_inputs
        if candidate_id not in bad_input
    ]

    old_scores_g = (
        previous_scores
        .reindex(good_input)
        .to_numpy()
    )

    old_scores_b = (
        previous_scores
        .reindex(bad_input)
        .to_numpy()
    )

    old_scores_rank_g = (
        previous_ranks
        .reindex(good_input)
        .to_numpy()
    )

    old_scores_rank_b = (
        previous_ranks
        .reindex(bad_input)
        .to_numpy()
    )
    
    good_titles = (
        db.loc[
            db[id_col].isin(good_inputs),
            title_col
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    bad_titles = (
        db.loc[
            db[id_col].isin(bad_inputs),
            title_col
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    db_temp = db

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
            ce_weight=0.70,
            tfidf_weight=0.30
        )

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
            ce_weight=0.70,
            tfidf_weight=0.30
        )

    if "final_bad_score_max" in db_temp.columns:
        bad_avoidance = (1 - db_temp["final_bad_score_max"])

    weighted_score = (
        0.5
        * db_temp[base_col]
    )

    active_weight = 0.5

    if "final_good_score_max" in db_temp.columns:
        weighted_score += (
            0.3
            * db_temp["final_good_score_max"]
        )
        active_weight += 0.3

    if "final_bad_score_max" in db_temp.columns:
        bad_avoidance = (
            1
            - db_temp["final_bad_score_max"]
        )

        weighted_score += (
            0.2
            * bad_avoidance
        )
        active_weight += 0.2

    db_temp["model_fit_score"] = (
        weighted_score
        / active_weight
    )

    db_temp["model_rank"] = db_temp[
        "model_fit_score"
    ].rank(
        ascending=False,
        method="min"
    )

    indexed_temp = db_temp.set_index(id_col)

    new_scores_g = (
        indexed_temp
        .reindex(good_input)["model_fit_score"]
        .to_numpy()
    )

    new_scores_b = (
        indexed_temp
        .reindex(bad_input)["model_fit_score"]
        .to_numpy()
    )

    new_scores_rank_g = (
        indexed_temp
        .reindex(good_input)["model_rank"]
        .to_numpy()
    )

    new_scores_rank_b = (
        indexed_temp
        .reindex(bad_input)["model_rank"]
        .to_numpy()
    )

    db_temp["final_fit_with_weights_mean"] = (
        db_temp["model_fit_score"].copy()
    )

    change_counter += 1
    feedback_metrics_history = (
        update_feedback_metrics(
            metrics_table=feedback_metrics_history,
            db=db_temp,
            id_col=id_col,
            score_col="model_fit_score",
            good_ids=good_inputs,
            bad_ids=bad_inputs,
            change_number=change_counter,
            k=10
        )
    )

    db_temp.loc[
        db_temp[id_col].isin(good_inputs),
        "final_fit_with_weights_mean"
    ] = 1.0

    db_temp.loc[
        db_temp[id_col].isin(bad_inputs),
        "final_fit_with_weights_mean"
    ] = 0.0

    db_temp["rank"] = db_temp["final_fit_with_weights_mean"].rank(ascending=False, method="min")

    db = db_temp.sort_values(["rank"], 
            ascending=[True])
    
    if good_input:
        ranking_history = update_ranking_history(change_counter, 
                                                ranking_history,
                                                good_input,
                                                "Good",
                                                old_scores_g,
                                                new_scores_g,
                                                old_scores_rank_g,
                                                new_scores_rank_g)

    if bad_input:
        ranking_history = update_ranking_history(change_counter, 
                                                    ranking_history,
                                                    bad_input,
                                                    "Bad",
                                                    old_scores_b,
                                                    new_scores_b,
                                                    old_scores_rank_b,
                                                    new_scores_rank_b)
        
    return(db)


def add_initial_relevance_gate(
    db,
    ce_column=(
        "final_ce_base_score_mean"
    ),
    tfidf_column=(
        "final_tfidf_base_score_max"
    ),
    semantic_floor_quantile=0.40
):
    semantic_floor = db[
        ce_column
    ].quantile(
        semantic_floor_quantile
    )

    db["passes_initial_gate"] = (
        (
            db[ce_column]
            >= semantic_floor
        )
        |
        (
            db[tfidf_column] > 0
        )
    )

    return db, semantic_floor

my_query = new_query()
# Semantic base scores
data = fit_model_to_query(
    my_query,
    "job_title",
    ce_model,
    data,
    "ce_base"
)

# Lexical base scores
data = fit_tfidf_to_query(
    my_query,
    "job_title",
    data,
    "tfidf_base"
)

# Combined base scores
data = combine_similarity_scores(
    data,
    ce_desc="ce_base",
    tfidf_desc="tfidf_base",
    output_desc="base",
    ce_weight=0.70,
    tfidf_weight=0.30
)



data["rank"] = data[
    "final_base_score_mean"
].rank(
    ascending=False,
    method="min"
)

data = data.sort_values(
    [
        "rank"
    ],
    ascending=True
)

data, initial_semantic_floor = (
    add_initial_relevance_gate(data)
)

data = reranking_algo(data, "id", "job_title", "final_base_score_mean", ce_model)

end_reranking = 0

while end_reranking == 0:

    want_rerank = input("Do you want to continue adding changes? Type Y if so.")

    if want_rerank.strip().casefold() in ["y", "yes"]:
        data = reranking_algo(data, "id", "job_title", "final_base_score_mean", ce_model)

    else:
        end_reranking = 1


data.to_csv('trial_ranking.csv', index=False)
ranking_history.to_csv("ranking_history.csv",index=False)

### HAS STAR

# When we have a star, we want the scores to update dynamically. Such that, the fit becomes 1 for starred entries,
# and the final score gets updated to compare with the starred entries.

print(ranking_history)

feedback_metrics_history.to_csv(
    "feedback_metrics_history.csv",
    index=False
)

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
print(initially_excluded)
print(feedback_metrics_history)