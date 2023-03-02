import pickle


def top_n_songs(final_play_counts, n, min_playbacks):
    recommendations = final_play_counts[final_play_counts['play_freq'] > min_playbacks]
    recommendations = recommendations.sort_values(by='avg_count', ascending=False)
    return recommendations.index[:n]


def get_song_titles(song_ids):
    titles = []
    id_lookup = pickle.load(open('../music-recommendations/Models/title_dictionary.pkl', 'rb'))
    for song_id in song_ids:
        titles.append(id_lookup[song_id])
    return titles
