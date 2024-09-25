from elasticsearch import Elasticsearch
from typing import Dict, List

class ElasticsearchClient:
    def __init__(self, url: str):
        """
        Initialize the Elasticsearch client.

        Args:
            url (str): URL of the Elasticsearch instance.
        """
        self.client = Elasticsearch(url)

    def create_index(self, index_name: str, index_settings: Dict):
        """
        Create an Elasticsearch index.

        Args:
            index_name (str): Name of the index to create.
            index_settings (Dict): Settings and mappings for the index.
        """
        self.client.indices.delete(index=index_name, ignore_unavailable=True)
        self.client.indices.create(index=index_name, body=index_settings)

    def index_document(self, index_name: str, document: Dict):
        """
        Index a document in Elasticsearch.

        Args:
            index_name (str): Name of the index to use.
            document (Dict): Document to index.
        """
        self.client.index(index=index_name, document=document)

    def knn_search(self, index_name: str, field: str, vector: List[float], k: int = 5) -> List[Dict]:
        """
        Perform a k-NN search in Elasticsearch.

        Args:
            index_name (str): Name of the index to search.
            field (str): Field to search in.
            vector (List[float]): Query vector.
            k (int): Number of nearest neighbors to return.

        Returns:
            List[Dict]: List of search results.
        """
        knn = {
            "field": field,
            "query_vector": vector,
            "k": k,
            "num_candidates": 10000
        }
        search_query = {
            "knn": knn,
            "_source": ["question", "answer"]
        }
        es_results = self.client.search(
            index=index_name,
            body=search_query
        )
        
        return [hit['_source'] for hit in es_results['hits']['hits']]