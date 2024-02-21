from fastapi import FastAPI
from routers import chat
from routers import rag

#app = FastAPI(dependencies=[Depends(get_query_token)])
app = FastAPI()

app.include_router(chat.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    return {"message": "open-ai service apis active..."}