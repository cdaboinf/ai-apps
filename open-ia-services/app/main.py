#from fastapi import Depends, FastAPI
from fastapi import FastAPI

#from .dependencies import get_query_token, get_token_header
#from .internal import admin
from routers import chat

#app = FastAPI(dependencies=[Depends(get_query_token)])
app = FastAPI()


app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "open-ai service apis active..."}