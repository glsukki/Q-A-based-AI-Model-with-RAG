import os
import uuid
import torch
import requests
import transformers

import numpy as np
import pandas as pd

from openai import OpenAI
from typing import Optional
from .vector_db import DataEmbedding
from .prompts import SYSTEM_PROMPT, USER_PROMPT

class Pipeline:
    def __init__(self, data_path):
        self.open_ai_key = os.environ["OPENAI_API_KEY"]
        self.model_name = "gpt-4o"
        self.data_path = data_path
        self.db_embedding_obj = DataEmbedding()

    def generative_model(self, user_query, context):
        client = OpenAI(
            api_key=self.open_ai_key
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(query=user_query, context=context)}
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content

    def retrieve_top_contexts(self, user_query, top_k_documents, similarity_measure: Optional[str] = "cosine_similarity"):
        self.blog_data_path = self.data_path + "/blog.json"        
        ## Read in the json data
        blog_data = self.db_embedding_obj.read_data(data_path=self.blog_data_path)
        ## Insert into vector db
        self.db_embedding_obj.insert_vdb(data=blog_data)

        ## Obtain the top k relevant documents/contextual information relevant to the given user query
        top_k_results = self.db_embedding_obj.data_embedding(
            user_query=user_query,
            top_k_documents=top_k_documents,
            similarity_measure=similarity_measure
        )
        
        top_k_blogs = []
        for result in top_k_results:
            top_k_blogs.append(result["entry"]["metadata"]["text"])
        
        lm_response = self.generative_model(
            user_query=user_query,
            context=top_k_blogs
        )
        # print(f"LM Response to Query: \n{lm_response}")
        return lm_response

    # ## TODO: Fix the below method
    # def manual_insert_and_retrieve_top_contexts(self, index, data, user_query):
    #     data = {
    #         "id": uuid.uuid4(),
    #         "metadata": data
    #     }
    #     self.db_embedding_obj.create_indices(index)
    #     self.db_embedding_obj.insert_vdb(data)
    #     ## Obtain the top k relevant documents/contextual information relevant to the given user query
    #     top_k_results = self.db_embedding_obj.data_embedding(user_query=user_query, similarity_measure="euclidean")
        
    #     top_k_blogs = []
    #     for result in top_k_results:
    #         top_k_blogs.append(result["entry"]["metadata"]["text"])
        
    #     lm_response = self.generative_model(
    #         user_query=user_query,
    #         context=top_k_blogs
    #     )
    #     print(f"LM Response to Query: \n{lm_response}")