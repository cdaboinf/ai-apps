from models.chat_request import chat_request
from typing import TypeVar, List
from abc import ABC, abstractmethod

class IChatService(ABC):

    T = TypeVar('T')
    @abstractmethod 
    def process_chat(self, request: chat_request) -> List[T]:
        """process LLM chat request."""
        pass