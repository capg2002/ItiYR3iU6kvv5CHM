from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np

data = pd.read_csv("filtered_talents_db.csv")
data["id"] = data["id"].astype(str)

rank_col_names = ['Change Number', 'ID(s)', 'Change Type', 'Old Score', 'Updated Score',
                  'Old Rank', 'New Rank']
ranking_history = pd.DataFrame(columns=rank_col_names)

# Load a pre-trained CrossEncoder model
ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

    new_data = pd.DataFrame({'Change Number': changelog,
                             'ID(s)': updated_input, 
                             'Change Type': change_type,
                             'Old Score': old_score, 
                             'Updated Score': new_score,
                             'Old Rank': old_rank,
                             'New Rank': new_rank,})
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
        if column.startswith(f"score_{desc}")
    ]
    db[f"final_{desc}_score_mean"] = db[score_columns].mean(axis=1)
    db[f"final_{desc}_score_sum"] = db[score_columns].sum(axis=1)
    db[f"final_{desc}_score_max"] = db[score_columns].max(axis=1)
    db[f"final_{desc}_score_min"] = db[score_columns].min(axis=1)

    return(db)

def fit_model_to_query(query_input, title_col, model, db, desc):
    # Predict scores for a pair of sentences
    for query in query_input:
        print("THIS IS CURRENT QUERY:", query)
        pairs = [[query, text] for text in db[title_col]]
        scores = model.predict(pairs)

        # Min-max normalization to produce scores from 0 to 1.
        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            normalized_scores = np.zeros_like(scores)
        else:
            normalized_scores = (
                (scores - minimum)
                / (maximum - minimum)
            )

        column_name = (
            f"score_{desc}_{normalize_for_column(query)}"
        )

        db[column_name] = normalized_scores

    if len(query_input) > 0:

        db = make_final_scores(db, desc)

        db = db.sort_values([f"final_{desc}_score_mean"], ascending=False)
        db["rank"] = db["final_{desc}_score_mean"].rank(ascending=False, method="min")


    return(db)

def add_fit_inputs(db, id_col, fit_type):
    decision_input = input(
        f"Would you like to add {fit_type} fit indices? Type Y if yes. "
    )

    if decision_input == "Y":
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

def reranking_algo(db, id_col, title_col, base_col, ce_model):
    global good_inputs
    global bad_inputs
    global change_counter
    global ranking_history

    feedback_columns = [
    column
    for column in db.columns
        if column.startswith((
            "score_good_",
            "score_bad_",
            "final_good_",
            "final_bad_",
            "final_fit_"
        ))
    ]

    db = db.drop(
        columns=feedback_columns,
        errors="ignore"
    )

    remove_inputs = input("Would you like to remove any conditions? Type Y if so.")
    if remove_inputs == "Y":
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

    if "final_fit_with_weights_mean" in db:
        old_scores_g = db.loc[db[id_col].isin(good_input), "final_fit_with_weights_mean"]
        old_scores_b = db.loc[db[id_col].isin(bad_input), "final_fit_with_weights_mean"]
    else:
        old_scores_g = db.loc[db[id_col].isin(good_input), "final_base_score_mean"]
        old_scores_b = db.loc[db[id_col].isin(bad_input), "final_base_score_mean"]

    old_scores_rank_g = db.loc[db[id_col].isin(good_input), "rank"]
    old_scores_rank_b = db.loc[db[id_col].isin(bad_input), "rank"]
    

    good_titles = db.loc[
        db[id_col].isin(good_inputs),
        title_col
    ]

    bad_titles = db.loc[
        db[id_col].isin(bad_inputs),
        title_col
    ]

    db_temp = fit_model_to_query(good_titles, title_col, ce_model, db, "good")
    db_temp = fit_model_to_query(bad_titles, title_col, ce_model, db_temp, "bad")

    if "final_bad_score_max" in db_temp.columns:
        bad_avoidance = (1 - db_temp["final_bad_score_max"])

    weighted_score = (
        0.5
        * db_temp["final_base_score_mean"]
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

    db_temp["final_fit_with_weights_mean"] = (
        weighted_score
        / active_weight
    )

    new_scores_g = db_temp.loc[db_temp[id_col].isin(good_input), "final_fit_with_weights_mean"]
    new_scores_b = db_temp.loc[db_temp[id_col].isin(bad_input), "final_fit_with_weights_mean"]

    change_counter += 1


    db_temp.loc[
        db_temp[id_col].isin(good_inputs),
        "final_fit_with_weights_mean"
    ] = 1.0

    db_temp.loc[
        db_temp[id_col].isin(bad_inputs),
        "final_fit_with_weights_mean"
    ] = 0.0

    db = db_temp.sort_values(["final_fit_with_weights_mean"], 
                ascending=[False])

    db["rank"] = db["final_fit_with_weights_mean"].rank(ascending=False, method="min")

    new_scores_rank_g = db.loc[db[id_col].isin(good_input), "rank"]
    new_scores_rank_b = db.loc[db[id_col].isin(bad_input), "rank"]
    
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


my_query = new_query()
data = fit_model_to_query(my_query, "job_title", ce_model, data, "base")

data = reranking_algo(data, "id", "job_title", "final_base_score_mean", ce_model)

end_reranking = 0

while end_reranking == 0:

    want_rerank = input("Do you want to continue adding changes? Type Y if so.").strip().casefold()

    if want_rerank in ["y", "yes"]:
        data = reranking_algo(data, "id", "job_title", "final_base_score_mean", ce_model)

    else:
        end_reranking = 1


data.to_csv('trial_ranking.csv', index=False)
ranking_history.to_csv("ranking_history.csv",index=False)

### HAS STAR

# When we have a star, we want the scores to update dynamically. Such that, the fit becomes 1 for starred entries,
# and the final score gets updated to compare with the starred entries.

print(ranking_history)