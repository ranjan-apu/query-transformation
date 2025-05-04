# app/techniques/query_decomposition.py
from typing import List
from langchain_core.documents import Document

from src.app.techniques.base import BaseRAGTechniques


class QueryDecomposition(BaseRAGTechniques):
    def retrieve(self, query: str) -> List[Document]:
        # Decompose the query into sub-queries
        sub_queries = self._decompose_query(query)

        # Retrieve documents for each sub-query
        all_docs = []
        doc_ids = set()

        for sub_query in sub_queries:
            docs = self.db.similarity_search(sub_query)

            # Add only unique documents
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in doc_ids:
                    all_docs.append(doc)
                    doc_ids.add(doc_id)

        return all_docs

    def _decompose_query(self, query: str) -> List[str]:
        prompt = f"""Break down this complex query into 2-4 simpler sub-queries that together cover all aspects.

Complex query: {query}

Sub-queries:"""

        response = self.llm.invoke(prompt).content

        # Parse the response to extract sub-queries
        sub_queries = []
        for line in response.strip().split('\n'):
            if line and not line.startswith('Sub-queries:'):
                clean_line = line.lstrip('0123456789. -')
                if clean_line:
                    sub_queries.append(clean_line)

        return sub_queries if sub_queries else [query]
