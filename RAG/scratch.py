import os
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from groq import Groq

# Initialize model and Elasticsearch client
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
es_client = Elasticsearch('http://localhost:9200')

def elastic_search_knn(field, vector, index_name="customer-support"):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000
    }
    search_query = {
        "knn": knn,
        "_source": ["question", "answer"]
    }
    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    return [hit['_source'] for hit in es_results['hits']['hits']]

def question_answer_vector_knn(question):
    v_q = model.encode(question)
    return elastic_search_knn('question_answer_vector', v_q)

def build_prompt(query, search_results):
    prompt_template = """
You are a highly knowledgeable, friendly, and empathetic and helpful customer support assistant for a telecommunications company.
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

def llm(prompt, model='llama-3.2-90b-text-p'):
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query, model='llama-3.1-70b-versatile') -> str:
    question = query if isinstance(query, str) else query.get('Question') or query.get('question')
    if not question:
        raise ValueError("Input must be a string or a dictionary with a 'Question' or 'question' key")
    
    search_results = question_answer_vector_knn(question)
    prompt = build_prompt(question, search_results)
    answer = llm(prompt, model=model)
    return answer

def get_answer_for_question(question):
    return rag(question)

if __name__ == "__main__":
    # Set up the Groq API key
    os.environ["GROQ_API_KEY"] = "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"
    
    custom_question = input("Enter your question: ")
    answer = get_answer_for_question(custom_question)
    print(f"\nQuestion: {custom_question}")
    print(f"Answer: {answer}")