from fastapi import APIRouter, UploadFile, Depends
from services.chat_service import chat_service
from models.chat_request import chat_request

router = APIRouter()

def get_chat_service():
    return chat_service()

@router.post("/chat/context/upload", tags=["AI Chat"])
async def upload(file: UploadFile, service: chat_service = Depends(get_chat_service)):  
    filename = service.upload_rag_document_context(file)

    return {"file": filename}

@router.post("/chat/rag", tags=["AI Chat"])
async def search(request: chat_request, service: chat_service = Depends(get_chat_service)):
    chat_messages = service.process_rag_chat(request)

    return chat_messages

@router.delete("/chat/context/clear", tags=["AI Chat"])
async def clear_collection(service: chat_service = Depends(get_chat_service)):
    service.clear_document_collection()