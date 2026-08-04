from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np

data = pd.read_csv("filtered_talents_db.csv")
data["id"] = data["id"].astype(str)

rank_col_names = ['ID(s)', 'Old Score', 'Updated Score']
ranking_history = pd.DataFrame(columns=rank_col_names)

# Load a pre-trained CrossEncoder model
ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
good_inputs = []
bad_inputs = []

def update_ranking_history(rank_table, old_score, new_score, updated_input):
    new_data = pd.DataFrame({'ID(s)': updated_input, 'Old Score': old_score, 'Updated Score': new_score})
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
            f"score_{desc}_{query.lower().replace(' ', '_')[:30]}"
        )

        db[column_name] = normalized_scores
        db = make_final_scores(db, desc)


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

        while not db[id_col].isin(last_input).any():
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

def reranking_algo(db, id_col, title_col, ce_model):


    good_input = add_fit_inputs(db, id_col, "good")

    good_titles = db.loc[db[id_col].isin(good_input), title_col]
    bad_input = add_fit_inputs(db, id_col, "bad")
    bad_titles = db.loc[db[id_col].isin(bad_input), title_col]

    good_inputs += good_titles
    bad_inputs += bad_titles

    db_temp = fit_model_to_query(good_titles, title_col, ce_model, db, "good")
    db_temp = fit_model_to_query(bad_titles, title_col, ce_model, db_temp, "bad")

    db_temp["final_fit_with_weights_max"] = 0.7*db_temp["final_base_score_max"] + 0.3*db_temp["final_good_score_max"] - 0.2*db_temp["final_bad_score_max"]
    db_temp["final_fit_with_weights_mean"] = 0.7*db_temp["final_base_score_mean"] + 0.3*db_temp["final_good_score_mean"] - 0.2*db_temp["final_bad_score_mean"]

    db = db_temp.sort_values(["final_fit_with_weights_mean","final_fit_with_weights_max"], 
                ascending=[False, False])
    return(db)


my_query = new_query()
data = fit_model_to_query(my_query, "job_title", ce_model, data, "base")

data = reranking_algo(data, "id", "job_title", ce_model)

data.to_csv('trial_ranking.csv', index=False)

### HAS STAR

# When we have a star, we want the scores to update dynamically. Such that, the fit becomes 1 for starred entries,
# and the final score gets updated to compare with the starred entries.