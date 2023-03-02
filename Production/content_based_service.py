import Models.content_based as cb
import pickle


def content_based_recommendations(name: str, n: int):
    df = pickle.load(open('../music-recommendations/Models/songs_with_text_feature.pkl', 'rb'))
    similar_songs = pickle.load(open('../music-recommendations/Models/tfidf_similarities.pkl', 'rb'))
    return cb.recommendations(df, name, similar_songs, n)
