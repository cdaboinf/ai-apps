from fastapi import APIRouter, UploadFile
from services.rag_service import rag_service
from models.rag_request import rag_request

router = APIRouter()

@router.post("/rag/upload", tags=["AI RAG"])
async def upload(file: UploadFile):  
    service = rag_service()
    filename = service.upload_document(file)

    return {"file": filename}

@router.post("/rag/search", tags=["AI RAG"])
async def search(request: rag_request):
    service = rag_service()
    search_result = service.search(request)

    return search_result

@router.delete("/rag/clear", tags=["AI RAG"])
async def clear_collection():
    service = rag_service()
    service.clear_document()