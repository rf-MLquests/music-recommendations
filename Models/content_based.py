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

df_final = pd.read_csv("../Data/playbacks.csv")
df_final['text'] = df_final['title'] + ' ' + df_final['release'] + ' ' + df_final['artist_name']
df_final = df_final[['user_id', 'song_id', 'play_count', 'title', 'text']]
df_final = df_final.drop_duplicates(subset=['title'])
df_final = df_final.set_index('title')


def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stopwords.words('english')]  # Use stopwords of english
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]
    return text_lems


def build_song_tfidf():
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    song_tfidf = tfidf.fit_transform(df_final['text'].values).toarray()
    return song_tfidf


def recommendations(title, similar_songs):
    recommended_songs = []
    indices = pd.Series(df_final.index)
    idx = indices[indices == title].index[0]
    score_series = pd.Series(similar_songs[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1: 11].index)
    print(top_10_indexes)
    for i in top_10_indexes:
        recommended_songs.append(list(df_final.index)[i])
    return recommended_songs


def compute_similarities(song_tfidf):
    return cosine_similarity(song_tfidf, song_tfidf)


print(recommendations("Learn To Fly", compute_similarities(build_song_tfidf())))
