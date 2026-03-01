# MovieMatch

MovieMatch is a small recommender systems project built around MovieLens 100K. A recruiter can search movies, rate a short list of titles, and get personalized recommendations in a Streamlit app. The offline training step learns movie embeddings from historical positive interactions, and the app infers a brand-new user profile from the movies rated during the session.

![MovieMatch screenshot](assets/streamlit-screenshot.png)

## How It Works

1. Download MovieLens 100K from GroupLens into a local cache.
2. Convert historical ratings into implicit feedback by treating ratings `>= 4` as positive interactions.
3. Train an implicit ALS model to learn user and movie embeddings.
4. When a new user rates movies in Streamlit, keep the movie embeddings fixed and solve a small ridge regression problem to infer that user’s vector.
5. Score unseen movies with a dot product and show simple explanations based on the rated movies that are closest in embedding space.

If `implicit` is not available, the project still supports a popularity baseline so the app and evaluation flow remain usable.

## Project Layout

```text
movierec-interactive/
  movierec/
  app/
  scripts/
  tests/
  artifacts/
  assets/
```

## Quickstart

From the `movierec-interactive/` directory:

```bash
pip install -e .
python scripts/download_data.py
python scripts/train_model.py
streamlit run app/streamlit_app.py
```

## Evaluation

Run:

```bash
python scripts/run_eval.py
```

The script compares the personalized ALS model against a simple popularity baseline and prints a compact table. In plain English, it answers three questions:
- How often does the model put a held-out liked movie in the top 10 suggestions?
- How high in the list do those good suggestions appear?
- How much of the movie catalog actually shows up in recommendations?

Example output format:

```text
Model         Hits in Top 10   Ranks Good Picks Higher   Catalog Spread
----------------------------------------------------------------------
ALS                    0.1284                   0.0821           0.4174
Popularity             0.0637                   0.0362           0.0589
```

## Streamlit App Features

- Search movies by title substring.
- Add, update, and remove 1-5 star ratings in session state.
- Choose `ALS` or a popularity baseline.
- Request 5-20 recommendations.
- Inspect similar movies for any title in the catalog.

## Deploy on Streamlit Community Cloud

1. Push this repository to a public GitHub repo.
2. Make sure `pyproject.toml` is committed so Streamlit can install dependencies.
3. In Streamlit Community Cloud, create a new app from the GitHub repository.
4. Set the entrypoint to `app/streamlit_app.py`.
5. Run `python scripts/download_data.py` and `python scripts/train_model.py` locally once so `artifacts/model_artifacts.pkl` exists.
6. Commit `artifacts/model_artifacts.pkl` to GitHub, but do not commit `artifacts/data/`.
7. Deploy without secrets; this project does not require API keys.

## Notes

- The MovieLens dataset is not committed to the repository.
- The trained model file `artifacts/model_artifacts.pkl` can be committed for Streamlit deployment.
- Downloaded dataset files under `artifacts/data/` should stay out of git.
- The recommendation explanations are intentionally simple and honest: they point to rated movies whose learned vectors are closest to the recommended movie.

## Testing

```bash
pytest
```

If `pytest` is not installed in your environment yet, install the dev dependency set or run `python -m pytest` after installing dependencies.

## Attribution

This project uses the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset from GroupLens Research at the University of Minnesota. The dataset is downloaded locally by script and is not included in this repository.
