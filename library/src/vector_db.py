import json
import numpy as np
from enum import Enum
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMeasure(str, Enum):
    COSINE = "cosine_similarity"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"

class Index:
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.entries = []
        self.model = SentenceTransformer("TaylorAI/bge-micro")
        
    def insert(self, vector: List[float], entry: Dict):
        self.vectors.append(vector)
        self.entries.append(entry)
    
    def search(self, query_vector: List[float], similarity_measure: Optional[str] = "cosine_similarity", k: int = 3) -> List[Dict]:
        if not self.vectors:
            return []

        vectors_array = np.array(self.vectors)
        query_array = np.array(query_vector).reshape(1, -1)
        
        # print(f"Looking for similarity through similarity measure: {similarity_measure}")
        
        if similarity_measure == SimilarityMeasure.COSINE:
            similarities = cosine_similarity(query_array, vectors_array)[0]
            top_k_indicies = np.argsort(similarities)[-k:][::-1]
        elif similarity_measure == SimilarityMeasure.DOT_PRODUCT:
            dot_products = np.dot(vectors_array, query_array.T).flatten()
            top_k_indicies = np.argsort(dot_products)[-k:][::-1]
            similarities = dot_products
        elif similarity_measure == SimilarityMeasure.EUCLIDEAN:
            distances = np.linalg.norm(vectors_array - query_array, axis = 1)
            top_k_indicies = np.argsort(distances)[:k]
            similarities = -distances
        
        return [
                {"similarity": similarities, "entry": self.entries[i]}
                for i in top_k_indicies
            ]

    def encode(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


class Database:
    def __init__(self):
        self.indices = {}

    def create_index(self, index_name: str):
        if index_name in self.indices:
            raise ValueError(f"Index '{index_name}' already exists")
        self.indices[index_name] = Index()

    def get_index(self, index_name: str) -> Index:
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' does not exist")
        return self.indices[index_name]

    def delete_index(self, index_name: str):
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' does not exist")
        del self.indices[index_name]

    def insert(self, index_name: str, vector: List[float], entry: Dict):
        index = self.get_index(index_name)
        index.insert(vector, entry)

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 1,
        measure: SimilarityMeasure = SimilarityMeasure.COSINE) -> List[Dict]:
        index = self.get_index(index_name)
        return index.search(query_vector, k, measure)
    

class DataEmbedding:
    def __init__(self):
        self.indexing_obj = Index()
        self.db_obj = Database()

    def read_data(self, data_path):
        with open(data_path, "r") as data_file:
            data = json.load(data_file)
        return data

    def create_indices(self, indexes):
        for indx in indexes:
            self.db_obj.create_index(indx)
    
    def insert_vdb(self, data):
        for entry in data:
            vector = self.indexing_obj.encode(entry["metadata"]["text"])
            self.indexing_obj.insert(vector=vector, entry=entry)

    def data_embedding(self, user_query, top_k_documents, similarity_measure: Optional[str] = "cosine_similarity"):
        query_vector = self.indexing_obj.encode(text=user_query)
        top_k_results = self.indexing_obj.search(query_vector=query_vector, similarity_measure=similarity_measure, k=top_k_documents)

        # print("Top 2 results:")
        # for result in top_k_results:
        #     result_similarity = result['similarity']
        #     print(f"Similarity: {result_similarity}")
        #     print(f"ID: {result['entry']['id']}")
        #     print(f"Text: {result['entry']['metadata']['text']}")
        #     print()

        return top_k_results
