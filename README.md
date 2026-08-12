# AI-Powered Talent Ranking and Interactive Reranking

This project develops an NLP-based candidate ranking system for talent sourcing. The goal is to rank candidate profiles by their relevance to a recruiter-defined search while allowing human feedback to iteratively change the ranking.

The original problem focuses on candidates related to the phrases **"Aspiring human resources"** and **"seeking human resources"**, but the ranking pipeline is designed so that the search query can be changed.

## Project Overview

The pipeline combines:

- semantic sentence similarity from a pretrained **CrossEncoder**
- lexical similarity from **TF-IDF + cosine similarity**
- geographic proximity using **geodesic distance**
- candidate **connection count**
- recruiter feedback through explicit **good-fit** and **bad-fit** examples
- a relevance gate and ranking diagnostics to evaluate the effect of feedback

The notebook version is intended to make the full workflow easier to inspect and rerun cell-by-cell.

## Repository Workflow

A recommended order for reviewing the project is:

1. `talent_ranking_notebook.ipynb` — end-to-end, mentor-friendly version of the workflow
2. `eda.py` — original exploratory preprocessing and geocoding script
3. `initial_ranking_model.py` — earlier ranking/reranking implementation
4. `initial_ranking_model copy.py` — newest script version with location and connection weighting
5. `geonorm_test.py` — isolated test of location standardization/geocoding
6. `geonorm_test_output.csv` — pre-geocoded dataset used by the notebook so geocoding does not need to be repeated immediately

## Data

The candidate data contains:

- `id` — unique candidate identifier
- `job_title` — candidate headline/job-title text
- `location` — candidate location
- `connection` — LinkedIn-style connection count
- `fit` — target fit score field supplied in the original problem, initially unlabelled

The raw dataset contains 104 rows. After removing duplicate combinations of job title, location, and connection count, the working dataset contains 53 unique candidate records.

## Methodology

### 1. Text and Data Preprocessing

Candidate job-title text is standardized before ranking.

The preprocessing workflow:

- removes duplicate candidate records based on `job_title`, `location`, and `connection`
- removes punctuation and non-letter characters from job titles
- converts text to lowercase
- removes English stop words
- converts `500+` connection values to a numeric value of `500`
- converts candidate IDs to strings for consistent user-feedback lookup

The EDA script also inspects character and n-gram frequencies to understand recurring wording in the candidate headlines.

### 2. Location Standardization

LinkedIn-style location strings are normalized before geocoding. For example:

- `Greater Boston Area` → `Boston`
- `Houston, Texas Area` → `Houston, Texas`

Unique cleaned locations are geocoded with **Nominatim** through `geopy`. A `RateLimiter` is used to avoid sending requests too quickly.

The resulting standardized address, latitude, and longitude are stored for each candidate.

To make the notebook easier to rerun, it uses the already-created `geonorm_test_output.csv` by default rather than geocoding every candidate on every run.

### 3. CrossEncoder Semantic Similarity

The semantic component uses:

```text
cross-encoder/stsb-roberta-base
```

Each recruiter query is paired with every candidate title. The CrossEncoder predicts sentence-pair similarity and the scores are clipped to the interval `[0, 1]`.

For multiple queries, both the mean and maximum similarity are retained.

This component captures semantic similarity even when the candidate title does not share the exact same words as the recruiter query.

### 4. TF-IDF and Cosine Similarity

A second relevance signal is generated using `TfidfVectorizer` with:

- lowercasing
- Unicode accent stripping
- unigrams and bigrams
- sublinear term frequency
- L2 normalization

Candidate titles and recruiter queries are fitted in a shared TF-IDF space, and cosine similarity is calculated between each candidate and each query.

As with the CrossEncoder, mean and maximum TF-IDF similarities are retained across multiple queries.

### 5. Baseline Hybrid Relevance Score

The current implementation combines three baseline signals:

- CrossEncoder semantic similarity
- TF-IDF lexical similarity
- location proximity

The configured scoring call uses:

- CrossEncoder weight: `0.70`
- TF-IDF weight: `0.30`
- location weight: `0.05`

The implementation divides by the total active weight, so these weights are normalized before the final baseline score is produced.

The CrossEncoder receives the largest weight because it can capture contextual/semantic relationships, while TF-IDF provides a transparent lexical-similarity signal. Location is deliberately a much smaller component.

### 6. Geographic Proximity

When a target job location is supplied, geodesic distance is calculated between the job coordinates and every candidate.

Distances are converted into a score from 0 to 1 using reversed min-max normalization:

```text
location_score = 1 - (distance - minimum_distance) / (maximum_distance - minimum_distance)
```

Therefore:

- candidates closer to the preferred location receive scores closer to `1`
- candidates farther away receive scores closer to `0`

If no target location is supplied in the notebook, a neutral location score is used so the rest of the pipeline can still run.

### 7. Initial Relevance Gate

A relevance gate is applied after baseline scoring.

The semantic floor is the **30th percentile** of the baseline CrossEncoder mean score.

A candidate passes the gate if either:

1. their CrossEncoder score is at or above the semantic floor, or
2. their maximum TF-IDF score is greater than zero

Candidates that do not pass are flagged rather than deleted. This preserves the profiles for human review while identifying candidates that appear weakly related to the original search.

### 8. Human-in-the-Loop Reranking

The system supports iterative recruiter feedback.

The recruiter can identify candidate IDs as:

- **good fits**
- **bad fits**

The titles belonging to these candidates become new feedback queries.

The same two NLP approaches are then applied again:

- CrossEncoder semantic similarity
- TF-IDF cosine similarity

This means candidates similar to accepted profiles receive a positive signal, while candidates similar to rejected profiles can be penalized.

For bad examples, the system calculates:

```text
bad_avoidance = 1 - bad_similarity
```

A candidate therefore receives a larger bad-avoidance score when they are dissimilar to the rejected examples.

### 9. Final Feedback-Adjusted Score

The newest script uses the following reranking components:

- baseline relevance: `45%`
- normalized connection count: `5%`
- similarity to good examples: `35%`
- avoidance of bad examples: `15%`

Connection count is normalized as:

```text
connection_score = connection / 500
```

The score is divided by the sum of the active component weights. This allows the model to continue working when good or bad feedback has not yet been supplied.

After the evaluation metrics are recorded:

- explicitly accepted candidate IDs are assigned a final fit score of `1.0`
- explicitly rejected candidate IDs are assigned a final fit score of `0.0`

This preserves the recruiter's direct decision in the displayed ranking.

## Evaluation Metrics

The project tracks feedback quality using several ranking metrics.

### Mean Good Model Rank

Average rank of candidates marked as good. Lower values are better.

### Mean Bad Model Rank

Average rank of candidates marked as bad. Higher values are better.

### Good Recall@10

The proportion of recruiter-approved candidates found in the top 10 model results.

```text
Good Recall@10 = approved candidates in top 10 / total approved candidates
```

### Good-Bad Pairwise Accuracy

Measures how often a good candidate receives a higher score than a bad candidate.

A value of `1.0` means every evaluated good candidate outranked every evaluated bad candidate.

### Good-Bad Score Margin

Difference between the mean score of good candidates and the mean score of bad candidates:

```text
mean(good scores) - mean(bad scores)
```

A positive margin indicates that the model is separating accepted and rejected examples in the desired direction.

## Development Results

The saved evaluation output from the development run contains one feedback update with:

| Metric | Result |
|---|---:|
| Good feedback examples | 2 |
| Bad feedback examples | 1 |
| Mean good model rank | 4.5 |
| Mean bad model rank | 34.0 |
| Good Recall@10 | 1.00 |
| Good-Bad Pairwise Accuracy | 1.00 |
| Good-Bad Score Margin | 0.3255 |

In that run:

- both approved examples were retrieved within the top 10
- the approved examples ranked substantially above the rejected example on average
- every good-vs-bad comparison was ordered correctly
- the mean good score exceeded the bad score by approximately `0.3255`

The saved ranked dataset contains 53 candidate records. The initial relevance gate flagged:

- **37 candidates as passing**
- **16 candidates as initially excluded/low-relevance**

### Important result-version note

The saved `feedback_metrics_history.csv` and `trial_ranking.csv` were produced before the newest location/connection changes were fully reflected in the exported results. Therefore, the numerical results above should be treated as results from the earlier feedback-ranking run, while the **Methodology** section documents the newest scoring logic.

The notebook is structured so the updated model can be rerun and new results can be exported after the location and connection components are finalized.

## Running the Notebook

### 1. Clone the repository

```bash
git clone https://github.com/capg2002/ItiYR3iU6kvv5CHM.git
cd ItiYR3iU6kvv5CHM
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install pandas numpy scikit-learn sentence-transformers geopy nltk jupyter
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Open:

```text
talent_ranking_notebook.ipynb
```

Then choose **Run All** or run the notebook section-by-section.

> The first CrossEncoder run may download pretrained model files and therefore requires internet access.

## Notebook Configuration

Near the top of the notebook, edit the configuration cell:

```python
SEARCH_QUERIES = [
    "Aspiring human resources",
    "seeking human resources",
]

POSITION_COORDS = None

GOOD_IDS = []
BAD_IDS = []
```

To use geographic preference, supply a latitude/longitude pair:

```python
POSITION_COORDS = (43.6532, -79.3832)
```

To test feedback reranking:

```python
GOOD_IDS = ["67", "84"]
BAD_IDS = ["88"]
```

Then rerun the baseline and feedback cells.

## Outputs

The original scripts can generate:

- `trial_ranking.csv` — complete candidate-level ranking dataset with model features and scores
- `ranking_history.csv` — changes in scores/ranks for user-labelled candidates
- `feedback_metrics_history.csv` — evaluation metrics after feedback rounds
- `only_value_and_rank.csv` — simplified candidate ID, title, and final rank output

## Current Implementation Note

The newest Python script currently refers to `connections` during reranking, while the processed dataset uses the column name `connection`.

The notebook corrects this by consistently using:

```python
data["connection"]
```

The same one-line change should be made in the script version before treating it as the canonical production implementation.

## Limitations and Next Steps

This is a prototype ranking system and should support—not replace—human recruiting judgment.

Important next steps include:

- rerunning the evaluation after the location and connection updates
- validating the ranking on a larger labelled set of recruiter judgments
- testing sensitivity to the current manually selected component weights
- adding repeated or cross-validated ranking evaluation where enough labels become available
- reviewing whether connection count should influence candidate fit and whether that signal introduces unwanted bias
- persisting feedback/ranking history across sessions
- separating configuration, reusable model functions, and user interface logic if the project is converted into a production application

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- sentence-transformers
- CrossEncoder / RoBERTa
- TF-IDF
- cosine similarity
- geopy
- Nominatim
- NLTK
- Jupyter Notebook
