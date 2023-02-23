import Models.content_based as cb
import pandas as pd


def content_based_recommendations(name: str, n: int):
    df = pd.read_csv("../music-recommendations/Data/playbacks.csv")
    df = cb.assemble_text_features(df)
    tfidf = cb.build_song_tfidf(df)
    similar_songs = cb.compute_similarities(tfidf)
    return cb.recommendations(df, name, similar_songs, n)
