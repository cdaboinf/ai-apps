from fastapi import APIRouter
from services.chat_service import chat_service
from models.chat_request import chat_request

import json

router = APIRouter()

@router.post("/chat/interactive", tags=["AI Chat"])
async def chat_interactive(request: chat_request):
    service = chat_service("gpt-3.5-turbo", "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")
    chat_messages = service.process_chat(request)

    return chat_messages