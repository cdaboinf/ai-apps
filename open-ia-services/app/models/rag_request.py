from pydantic import BaseModel

class rag_request(BaseModel):
    query: str = ''
    session: str = ''