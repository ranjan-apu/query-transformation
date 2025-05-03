from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class QdrantDB:
    def __init__(self,host="localhost",port=6333,collection_name="pdf_documents"):
        self.client = QdrantClient(host=host,port=port)
        self.collection_name = collection_name

        def create_collection(self,embeddings,vector_size=1536):
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                print(f"Collection '{self.collection_name}' created.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
            
            self.embeddings = embeddings

        def add_documents(self,documents:List[Document]):
            Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=f"http://localhost:6333",
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        def similarity_search(self,query:str):
            qdrant = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=self.collection_name,
            )
            return qdrant.similarity_search(query)


