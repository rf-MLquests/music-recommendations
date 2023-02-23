import Models.collaborative_filtering as cf
import pandas as pd


def deserialize_recommendations(result):
    return result.to_dict(orient='records')


def collaborative_filtering_recommendations(type: str, user: int, n: int):
    df = pd.read_csv("../music-recommendations/Data/playbacks.csv")
    model = cf.model_builder(type, df)
    recommendations = cf.get_recommendations(df, user, n, model)
    result = cf.ranking_songs(df, recommendations)
    return deserialize_recommendations(result)
