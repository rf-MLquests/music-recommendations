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
