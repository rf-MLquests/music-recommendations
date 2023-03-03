import pickle


def top_n_songs(final_play_counts, n, min_playbacks):
    recommendations = final_play_counts[final_play_counts['play_freq'] > min_playbacks]
    recommendations = recommendations.sort_values(by='avg_count', ascending=False)
    return recommendations.index[:n]


def get_song_titles(song_ids):
    titles = []
    title_lookup = pickle.load(open('../music-recommendations/Models/title_dictionary.pkl', 'rb'))
    for song_id in song_ids:
        titles.append(title_lookup[song_id])
    return titles


def get_top_ranked(n=10, min=50):
    final_play = pickle.load(open('../music-recommendations/Models/play_frequencies.pkl', 'rb'))
    return get_song_titles(top_n_songs(final_play, n, min))
