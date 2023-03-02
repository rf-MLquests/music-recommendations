import pandas as pd
import pickle
import nltk
import Models.collaborative_filtering as cf

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tally_average_playcounts(df):
    avg_count = df.groupby(by="song_id").mean()['play_count']
    play_freq = df.groupby(by="song_id").count()['play_count']
    final_play = pd.DataFrame({'avg_count': avg_count, 'play_freq': play_freq})
    pickle.dump(final_play, open('../music-recommendations/Models/play_frequencies.pkl', 'wb'))


def build_title_dictionary(df):
    id_lookup = pd.Series(df['title'].values, index=df['song_id']).to_dict()
    pickle.dump(id_lookup, open('../music-recommendations/Models/title_dictionary.pkl', 'wb'))


def assemble_text_features(df):
    df['text'] = df['title'] + ' ' + df['release'] + ' ' + df['artist_name']
    df = df[['user_id', 'song_id', 'play_count', 'title', 'text']]
    df = df.drop_duplicates(subset=['title'])
    df = df.set_index('title')
    return df


def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]
    return text_lems


def build_song_tfidf(df):
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    song_tfidf = tfidf.fit_transform(df['text'].values).toarray()
    return song_tfidf


def compute_similarities(song_tfidf):
    return cosine_similarity(song_tfidf, song_tfidf)


def train_rank_based(df):
    tally_average_playcounts(df)
    build_title_dictionary(df)


def train_content_based(df):
    tfidf_df = df.copy()
    tfidf_df = assemble_text_features(tfidf_df)
    pickle.dump(tfidf_df, open('../music-recommendations/Models/songs_with_text_feature.pkl', 'wb'))

    tfidf = build_song_tfidf(tfidf_df)
    similar_songs = compute_similarities(tfidf)
    pickle.dump(similar_songs, open('../music-recommendations/Models/tfidf_similarities.pkl', 'wb'))


def train_collaborative_filtering(df):
    train, test = cf.split_dataset(df, test_size=0.4, random_state=42)
    print('user-user model: ')
    user_user_model = cf.user_user_model(train, test)
    pickle.dump(user_user_model, open('../music-recommendations/Models/user_user.pkl', 'wb'))
    print('item-item model: ')
    item_item_model = cf.item_item_model(train, test)
    pickle.dump(item_item_model, open('../music-recommendations/Models/item_item.pkl', 'wb'))
    print('SVD model: ')
    svd_model = cf.matrix_factorization_model(train, test)
    pickle.dump(svd_model, open('../music-recommendations/Models/svd.pkl', 'wb'))
    print('clustering-based model: ')
    clustering_model = cf.clustering_model(train, test)
    pickle.dump(clustering_model, open('../music-recommendations/Models/clustering_based.pkl', 'wb'))


def main():
    df = pd.read_csv("../music-recommendations/Data/playbacks.csv")
    pickle.dump(df, open('../music-recommendations/Models/playbacks.pkl', 'wb'))
    train_rank_based(df)
    train_content_based(df)
    train_collaborative_filtering(df)


if __name__ == "__main__":
    main()
