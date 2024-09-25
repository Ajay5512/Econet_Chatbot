import json
import pandas as pd
from typing import List, Dict

def load_documents(file_path: str) -> List[Dict]:
    """
    Load documents from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List[Dict]: List of document dictionaries.
    """
    with open(file_path, 'r') as f:
        return json.load(f)
        
    
def load_ground_truth(file_path: str) -> List[Dict]:
    """
    Load ground truth data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[Dict]: List of ground truth dictionaries.
    """
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')