import Production.ranked_based_service as rs
import Production.content_based_service
import Production.collaborative_filtering_service
from fastapi import FastAPI

app = FastAPI()


@app.get("/most-popular")
async def get_most_popular(k: int = 10, min: int = 50):
    return rs.get_top_ranked(k, min)
