from __future__ import annotations

from pathlib import Path

import streamlit as st

from movierec.artifacts import MODEL_ARTIFACTS_FILENAME, load_model_artifacts
from movierec.recommend import recommend, similar_movies


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@st.cache_resource
def load_artifacts() -> dict:
    return load_model_artifacts(ARTIFACTS_DIR)


def initialize_state() -> None:
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}


def add_rating(movie_id: int, rating: int) -> None:
    st.session_state.ratings[int(movie_id)] = int(rating)


def remove_rating(movie_id: int) -> None:
    st.session_state.ratings.pop(int(movie_id), None)


def main() -> None:
    st.set_page_config(page_title="MovieMatch", layout="wide")
    st.title("MovieMatch")
    st.write(
        "Rate a handful of movies, infer a new user profile from the trained embeddings, "
        "and get personalized recommendations without retraining the model."
    )
    initialize_state()

    st.selectbox("Dataset", ["MovieLens 100K"], index=0, disabled=True)

    try:
        artifacts = load_artifacts()
    except FileNotFoundError:
        st.error(
            f"Missing `{MODEL_ARTIFACTS_FILENAME}` in `{ARTIFACTS_DIR}`. "
            "Run `python scripts/download_data.py` and `python scripts/train_model.py` first."
        )
        return

    movies = artifacts["movies"].sort_values("title").reset_index(drop=True)
    movie_lookup = dict(zip(movies["title"], movies["movie_id"], strict=False))

    left_col, right_col = st.columns([1.3, 1.0])

    with left_col:
        st.subheader("Search and Rate")
        query = st.text_input("Search movies by title", placeholder="Toy Story, Star Wars, Heat...")
        filtered = movies
        if query.strip():
            filtered = movies[movies["title"].str.contains(query, case=False, na=False)]

        options = filtered["title"].head(100).tolist()
        selected_title = st.selectbox("Matching titles", options=options if options else ["No matches"])
        selected_rating = st.slider("Your rating", min_value=1, max_value=5, value=4)
        if st.button("Add / update rating", type="primary", disabled=not options):
            add_rating(movie_lookup[selected_title], selected_rating)

        st.subheader("Similar Movies")
        similar_title = st.selectbox("Choose a movie", options=movies["title"].tolist(), index=0)
        similar_k = st.slider("Similar results", min_value=5, max_value=20, value=10)
        if similar_title:
            similar_results = similar_movies(artifacts, movie_lookup[similar_title], k=similar_k)
            for row in similar_results:
                st.write(f"- {row['title']} ({row['score']:.3f})")

    with right_col:
        st.subheader("My Ratings")
        st.caption(f"Rated movies: {len(st.session_state.ratings)}")

        if not st.session_state.ratings:
            st.info("Add 5-20 ratings to get stronger recommendations.")
        else:
            for movie_id, rating in sorted(st.session_state.ratings.items(), key=lambda item: artifacts["movie_titles"][item[0]]):
                row_col, remove_col = st.columns([4, 1])
                row_col.write(f"{artifacts['movie_titles'][movie_id]} - {'★' * rating}")
                if remove_col.button("Remove", key=f"remove-{movie_id}"):
                    remove_rating(movie_id)
                    st.rerun()

        model_name = st.selectbox(
            "Model",
            options=["als", "popularity"],
            format_func=lambda value: "ALS" if value == "als" else "Popularity baseline",
        )
        top_k = st.slider("Recommendations", min_value=5, max_value=20, value=10)
        if st.button("Recommend", use_container_width=True):
            if not st.session_state.ratings:
                st.warning("Add at least one rating before requesting recommendations.")
            else:
                rows = recommend(artifacts, st.session_state.ratings, model=model_name, k=top_k)
                st.subheader("Recommendations")
                for row in rows:
                    st.markdown(f"**{row['title']}**")
                    st.write(f"Predicted score: {row['score']:.3f}")
                    st.caption(row["explanation"])


if __name__ == "__main__":
    main()
