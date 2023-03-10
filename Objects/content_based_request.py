from typing import Union
from pydantic import BaseModel


class ContentBasedRequest(BaseModel):
    song_name: str
    n: Union[int, None] = 10
