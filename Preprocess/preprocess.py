import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


def label_encode(df, col_name):
    column_encoder = LabelEncoder()
    encoded = column_encoder.fit_transform(df[col_name])
    df[col_name] = encoded


def create_count_dictionary(df, col_name):
    items = df[col_name]
    record_count = defaultdict()
    for item in items:
        if item in record_count:
            record_count[item] += 1
        else:
            record_count[item] = 1
    return record_count


def get_filter_list(record_count, cutoff):
    to_remove = []
    for item_id, num_records in record_count.items():
        if num_records < cutoff:
            to_remove.append(item_id)
    return to_remove


def filter_by_id_list(df, col_name, to_remove):
    new_df = df.loc[~ df[col_name].isin(to_remove)]
    return new_df


count_df = pd.read_csv("../Data/count_data.csv")
song_df = pd.read_csv("../Data/song_data.csv")
df_merged = count_df.merge(song_df.drop_duplicates(), how='left', on='song_id')
df_merged.drop(columns=["Unnamed: 0"], inplace=True)
print(df_merged.info())

label_encode(df_merged, 'user_id')
label_encode(df_merged, 'song_id')

user_rating_counts = create_count_dictionary(df_merged, 'user_id')
song_rating_counts = create_count_dictionary(df_merged, 'song_id')
users_to_remove = get_filter_list(user_rating_counts, 90)
songs_to_remove = get_filter_list(song_rating_counts, 120)
df1 = filter_by_id_list(df_merged, 'user_id', users_to_remove)
df_final = filter_by_id_list(df1, 'song_id', songs_to_remove)
df_final.loc[df_final['play_count'] > 5, 'play_count'] = 5
print(df_final.info())
df_final.to_csv("../Data/playbacks.csv", header=True, index=False)
