import Models.rank_based as rb
import pandas as pd


def get_top_ranked(n=10, min=50):
    df = pd.read_csv("../Data/playbacks.csv")
    final_play = rb.tally_average_playcounts(df)
    return rb.get_song_titles(rb.top_n_songs(final_play, n, min), df)
