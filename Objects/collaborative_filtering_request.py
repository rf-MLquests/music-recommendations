from typing import Union
from pydantic import BaseModel


class CollaborativeFilteringRequest(BaseModel):
    type: str
    user: int
    n: Union[int, None] = 10
