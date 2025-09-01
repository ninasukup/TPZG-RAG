# ------------------------------------------------------------------------------
# helpers/schemas.py - Pydantic data models for request and response validation
# ------------------------------------------------------------------------------
"""
Defines Pydantic models used for validating and structuring data exchanged
between services. Includes schemas for messages and chat responses.
"""

from pydantic import BaseModel, Field, Extra
from typing import Optional, List, Dict, Any

# --------------------------------------------------------------
# 1) Incoming Payload models
# --------------------------------------------------------------
class Message(BaseModel):
    id: Optional[str] = None
    parentId: Optional[str] = None
    childrenIds: Optional[List[str]] = None
    role: Optional[str] = None
    content: Optional[str] = None
    model: Optional[str] = None
    modelName: Optional[str] = None
    modelIdx: Optional[int] = None
    userContext: Optional[Any] = None
    timestamp: Optional[int] = None
    done: Optional[bool] = None
    
    class Config:
        extra = Extra.allow  # ignore any extra fields in messages

class CompletionRequest(BaseModel):
    model: Optional[str] = Field(default="default-model")
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    type: Optional[str] = None
    stream: Optional[bool] = None

    # If you still want max_tokens, temperature, etc., keep them:
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

    class Config:
        extra = Extra.allow  # allow any additional fields (params, background_tasks, etc.)

# --------------------------------------------------------------
# 2) Chat Completion Response models
# --------------------------------------------------------------
class ChatMessageResponse(BaseModel):
    role: str
    content: str

class ChatChoiceResponse(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoiceResponse]
    usage: Optional[Dict[str, int]] = None