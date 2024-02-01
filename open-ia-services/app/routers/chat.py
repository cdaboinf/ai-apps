from fastapi import APIRouter
from openai import OpenAI

import json
import os
import requests

router = APIRouter()


@router.post("/chat/", tags=["AI Chat"])
async def chat():
    return {"username": "fakecurrentuser"}


@router.post("/chat/interactive", tags=["AI Chat"])
async def chat_interactive():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

    # Example OpenAI Python library request
    MODEL = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Knock knock."},
            {"role": "assistant", "content": "Who's there?"},
            {"role": "user", "content": "Orange."},
        ],
        temperature=0,
    )

    chat_model = json.loads(response.model_dump_json())
    return chat_model