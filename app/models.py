from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class SummarizeInput(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 50