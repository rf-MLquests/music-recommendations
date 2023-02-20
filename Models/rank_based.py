import pandas as pd


def tally_average_playcounts(df):
    avg_count = df.groupby(by="song_id").mean()['play_count']
    play_freq = df.groupby(by="song_id").count()['play_count']
    final_play = pd.DataFrame({'avg_count': avg_count, 'play_freq': play_freq})
    return final_play


def top_n_songs(final_play_counts, n, min_playbacks):
    recommendations = final_play_counts[final_play_counts['play_freq'] > min_playbacks]
    recommendations = recommendations.sort_values(by='avg_count', ascending=False)
    return recommendations.index[:n]


def get_song_titles(song_ids, df):
    titles = []
    id_lookup = pd.Series(df['title'].values, index=df['song_id']).to_dict()
    for song_id in song_ids:
        titles.append(id_lookup[song_id])
    return titles

# final_play = tally_average_playcounts(df_final)
# print(get_song_titles(top_n_songs(final_play, 10, 50), df_final))
