import streamlit as st
import pandas as pd
from helper import preprocess_genres, recommend_movie

# Load data
df = pd.read_csv('ratings.csv')
df2 = pd.read_csv('movies.csv')
merged_df = pd.merge(df, df2, on='movieId', how='inner')

# Preprocess
final_df, genre_df = preprocess_genres(merged_df)

# UI
st.title("🎬 Movie Recommendation System")

movie_list = final_df['title'].unique()
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movie(selected_movie, final_df, genre_df)
    st.subheader("Recommendations:")
    for rec in recommendations:
        st.write(f"👉 {rec}")
