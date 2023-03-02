import Models.rank_based as rb
import pickle


def get_top_ranked(n=10, min=50):
    final_play = pickle.load(open('../music-recommendations/Models/play_frequencies.pkl', 'rb'))
    return rb.get_song_titles(rb.top_n_songs(final_play, n, min))
