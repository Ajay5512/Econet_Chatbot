from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerModel:
    def __init__(self, model_name: str):
        """
        Initialize the SentenceTransformer model.

        Args:
            model_name (str): Name of the pre-trained model to use.
        """
        self.model = SentenceTransformer(model_name)
        

    def encode(self, text: str) -> List[float]:
        """
        Encode text into a vector representation.

        Args:
            text (str): Text to encode.

        Returns:
            List[float]: Vector representation of the text.
        """
        return self.model.encode(text).tolist()