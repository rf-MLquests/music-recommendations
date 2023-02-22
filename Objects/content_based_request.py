from typing import Union
from pydantic import BaseModel


class ContentBasedRequest(BaseModel):
    song_name: str
