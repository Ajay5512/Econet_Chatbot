from typing import List, Dict
from groq import Groq
from src.models.sentence_transformer import SentenceTransformerModel
from src.database.elasticsearch_client import ElasticsearchClient

class RAG:
    def __init__(self, model: SentenceTransformerModel, es_client: ElasticsearchClient, index_name: str):
        """Initialize the RAG system."""
        self.model = model
        self.es_client = es_client
        self.index_name = index_name

    def build_prompt(self, query: str, search_results: List[Dict]) -> str:
        """Build the prompt for the language model."""
        prompt_template = """
        You are a highly knowledgeable, friendly, and empathetic customer support assistant for a telecommunications company.
        Your role is to assist customers by answering their questions with accurate, clear, and concise information based on the CONTEXT provided. 
        Please respond in a conversational tone that makes the customer feel heard and understood and Be concise, professional, and empathetic in your responses. 
        Use only the facts from the CONTEXT when answering the customer's QUESTION.
        If the CONTEXT does not have the context to answer the QUESTION, gently suggest that the customer 
        reach out to a live support agent for further assistance.

        QUESTION: {question}

        CONTEXT: 
        {context}
        """.strip()
        
        context = "\n\n".join([f"question: {doc['question']}\nanswer: {doc['answer']}" for doc in search_results])
        
        return prompt_template.format(question=query, context=context).strip()

    def llm(self, prompt: str, model: str = 'llama-3.2-90b-text-p') -> str:
        """Query the language model."""
        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    def get_answer(self, query: str, model: str = 'llama-3.1-70b-versatile') -> str:
        """Get an answer for a given query using the RAG system."""
        vector = self.model.encode(query)
        search_results = self.es_client.knn_search(self.index_name, 'question_answer_vector', vector)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt, model=model)

    