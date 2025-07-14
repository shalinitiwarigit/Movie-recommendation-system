import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def preprocess_genres(merged_df):
    sample_df = merged_df.drop_duplicates(subset='title').copy()
    sample_df['genre_list'] = sample_df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(sample_df['genre_list'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

    final_df = pd.concat([sample_df.drop(columns=['genres', 'genre_list']).reset_index(drop=True),
                          genre_df.reset_index(drop=True)], axis=1)

    return final_df, genre_df


def recommend_movie(movie_name, final_df, genre_df):
    movie_name = movie_name.strip().lower()
    matches = final_df[final_df['title'].str.strip().str.lower() == movie_name]

    if matches.empty:
        return [f"❌ Movie '{movie_name}' not found."]

    movie_index = matches.index[0]

    # Convert to sparse matrix
    genre_sparse = csr_matrix(genre_df.values)

    # Compute similarity only for this movie
    movie_vector = genre_sparse[movie_index]
    similarities = cosine_similarity(movie_vector, genre_sparse).flatten()

    similar_indices = similarities.argsort()[::-1][1:6]

    recommendations = []
    added_titles = set()
    for idx in similar_indices:
        title = final_df.iloc[idx]['title']
        if title not in added_titles:
            recommendations.append(title)
            added_titles.add(title)

    return recommendations
