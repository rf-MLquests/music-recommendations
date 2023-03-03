import pandas as pd
import numpy as np
import pickle


def model_loader(model_type: str):
    if model_type == "user":
        return pickle.load(open('../music-recommendations/Models/user_user.pkl', 'rb'))
    elif model_type == "item":
        return pickle.load(open('../music-recommendations/Models/item_item.pkl', 'rb'))
    elif model_type == "svd":
        return pickle.load(open('../music-recommendations/Models/svd.pkl', 'rb'))
    elif model_type == "clustering":
        return pickle.load(open('../music-recommendations/Models/clustering_based.pkl', 'rb'))
    else:
        raise ValueError("Invalid technique type")


def get_recommendations(user_id, top_n, algo):
    data = pickle.load(open('../music-recommendations/Models/playbacks.pkl', 'rb'))
    recommendations = []
    user_item_interactions_matrix = data.pivot_table(index='user_id', columns='song_id', values='play_count')
    non_interacted_products = user_item_interactions_matrix.loc[user_id][
        user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    for item_id in non_interacted_products:
        est = algo.predict(user_id, item_id).est
        recommendations.append((item_id, est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


def ranking_songs(recommendations):
    final_playbacks = pickle.load(open('../music-recommendations/Models/play_frequencies.pkl', 'rb'))
    ranked_songs = \
        final_playbacks.loc[[items[0] for items in recommendations]].sort_values('play_freq', ascending=False)[
            ['play_freq']].reset_index()

    ranked_songs = ranked_songs.merge(pd.DataFrame(recommendations, columns=['song_id', 'predicted_playcounts']),
                                      on='song_id', how='inner')

    ranked_songs['corrected_playcounts'] = ranked_songs['predicted_playcounts'] - 1 / np.sqrt(ranked_songs['play_freq'])

    ranked_songs = ranked_songs.sort_values(by='corrected_playcounts', ascending=False)
    return ranked_songs


def process_lookup(record):
    title_lookup = pickle.load(open('../music-recommendations/Models/title_dictionary.pkl', 'rb'))
    song_id = record['song_id']
    freq = record['play_freq']
    predicted = round(record['predicted_playcounts'], 2)
    corrected = round(record['corrected_playcounts'], 2)
    title = title_lookup[song_id]
    processed = {'title': title,
                 'timesPlayed': freq,
                 'predictedPlaycount': predicted,
                 'correctedPlaycount': corrected}
    return processed


def deserialize_recommendations(result):
    recommended = result.to_dict(orient='records')
    result_list = [process_lookup(rec) for rec in recommended]
    return result_list


def collaborative_filtering_recommendations(type: str, user: int, n: int):
    model = model_loader(type)
    recommendations = get_recommendations(user, n, model)
    result = ranking_songs(recommendations)
    return deserialize_recommendations(result)
