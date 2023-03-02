import Models.collaborative_filtering as cf


def deserialize_recommendations(result):
    return result.to_dict(orient='records')


def collaborative_filtering_recommendations(type: str, user: int, n: int):
    model = cf.model_loader(type)
    recommendations = cf.get_recommendations(user, n, model)
    result = cf.ranking_songs(recommendations)
    return deserialize_recommendations(result)
