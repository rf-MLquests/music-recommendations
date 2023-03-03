import pickle
import pandas as pd


def recommendations(df, title, similar_songs, n):
    recommended_songs = []
    indices = pd.Series(df.index)
    idx = indices[indices == title].index[0]
    score_series = pd.Series(similar_songs[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1: 1 + n].index)
    for i in top_10_indexes:
        recommended_songs.append(list(df.index)[i])
    return recommended_songs


def content_based_recommendations(name: str, n: int):
    df = pickle.load(open('../music-recommendations/Models/songs_with_text_feature.pkl', 'rb'))
    similar_songs = pickle.load(open('../music-recommendations/Models/tfidf_similarities.pkl', 'rb'))
    return recommendations(df, name, similar_songs, n)
