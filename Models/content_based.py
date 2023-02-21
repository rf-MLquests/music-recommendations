import pandas as pd
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def assemble_text_features(df):
    df['text'] = df['title'] + ' ' + df['release'] + ' ' + df['artist_name']
    df = df[['user_id', 'song_id', 'play_count', 'title', 'text']]
    df = df.drop_duplicates(subset=['title'])
    df = df.set_index('title')
    return df


def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]  # Use stopwords of english
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]
    return text_lems


def build_song_tfidf(df):
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    song_tfidf = tfidf.fit_transform(df['text'].values).toarray()
    return song_tfidf


def recommendations(df, title, similar_songs):
    recommended_songs = []
    indices = pd.Series(df.index)
    idx = indices[indices == title].index[0]
    score_series = pd.Series(similar_songs[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1: 11].index)
    print(top_10_indexes)
    for i in top_10_indexes:
        recommended_songs.append(list(df.index)[i])
    return recommended_songs


def compute_similarities(song_tfidf):
    return cosine_similarity(song_tfidf, song_tfidf)
