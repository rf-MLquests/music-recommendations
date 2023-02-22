import Production.ranked_based_service as rs
import Production.content_based_service as cb
import Production.collaborative_filtering_service as cf
from Objects.content_based_request import ContentBasedRequest
from fastapi import FastAPI

app = FastAPI()


@app.get("/mySpotify/most-popular")
async def get_most_popular(k: int = 10, min: int = 50):
    return rs.get_top_ranked(k, min)


@app.post("/mySpotify/content-based")
async def get_similar_songs(request: ContentBasedRequest):
    return cb.content_based_recommendations(request.song_name)
