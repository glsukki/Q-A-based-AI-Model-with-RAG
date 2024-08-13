import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from library.src.ai_pipeline import Pipeline

def main():
    data_path = os.path.abspath("../data/")
    obj = Pipeline(
        data_path=data_path
    )
    user_query = "What are the challenges in AI deployments"
    similarity_measure = "cosine_similarity"
    top_k_documents = 3
    rag_response = obj.retrieve_top_contexts(
        user_query=user_query,
        top_k_documents=top_k_documents,
        similarity_measure=similarity_measure
    )
    
    print(f"Response: \n{rag_response}")
    
    # manual_data = {
    #     "index": "Blog",
    #     "data": "This is an example on AI data"
    # }
    
    # obj.manual_insert_and_retrieve_top_contexts(
    #     index=manual_data["index"],
    #     data=manual_data["data"],
    #     user_query=user_query
    # )

if __name__ == "__main__":
    main()