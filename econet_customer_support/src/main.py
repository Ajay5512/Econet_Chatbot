from flask import Flask, render_template, request, jsonify
from src.data.loader import load_documents
from src.models.sentence_transformer import SentenceTransformerModel
from src.database.elasticsearch_client import ElasticsearchClient
from src.nlp.rag import RAG
from src.utils.env_config import load_env_vars
import os

app = Flask(__name__, template_folder='../ui/templates', static_folder='../ui/static')

# Load environment variables
load_env_vars()

# Initialize components
model = SentenceTransformerModel('multi-qa-MiniLM-L6-cos-v1')
es_client = ElasticsearchClient(os.getenv('ELASTICSEARCH_URL'))
rag_system = RAG(model, es_client, "customer-support")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    answer = rag_system.get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)