import os
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv

# Import the RAG functions
from rag import get_answer_for_question

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Pydantic models for request bodies
class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: int

# In-memory storage for conversation history (replace with database in production)
conversation_history: Dict[str, str] = {}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        logging.info(f"Received question: {request.question}")
        
        # Generate answer using RAG
        answer = get_answer_for_question(request.question)
        
        # Generate a new conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Store the conversation (in production, you'd use a database)
        conversation_history[conversation_id] = answer
        
        logging.info(f"Generated answer for conversation ID: {conversation_id}")
        
        return {"conversation_id": conversation_id, "answer": answer}
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your question")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        logging.info(f"Received feedback for conversation ID: {request.conversation_id}")
        
        if request.conversation_id not in conversation_history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if request.feedback not in [-1, 1]:
            raise HTTPException(status_code=400, detail="Invalid feedback value. Must be -1 or 1")
        
        # Here you would typically store the feedback in a database
        # For now, we'll just log it
        logging.info(f"Feedback received: {request.feedback} for conversation ID: {request.conversation_id}")
        
        return {"message": "Feedback received successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your feedback")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)