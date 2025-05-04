# app/techniques/step_back.py
from typing import List
from langchain_core.documents import Document

from src.app.techniques.base import BaseRAGTechniques


class StepBackPrompting(BaseRAGTechniques):
    def retrieve(self, query: str) -> List[Document]:
        # First, get the higher-level concept of the query
        higher_level_concept = self._identify_higher_level_concept(query)

        # Retrieve documents for both the original query and the higher-level concept
        concept_docs = self.db.similarity_search(higher_level_concept)
        specific_docs = self.db.similarity_search(query)

        # Combine and deduplicate results
        all_docs = []
        doc_ids = set()

        for doc_list in [concept_docs, specific_docs]:
            for doc in doc_list:
                doc_id = hash(doc.page_content)
                if doc_id not in doc_ids:
                    all_docs.append(doc)
                    doc_ids.add(doc_id)

        return all_docs

    def _identify_higher_level_concept(self, query: str) -> str:
        prompt = f"""Identify a higher-level, more general concept related to this question.

Specific question: {query}

Higher-level concept:"""

        response = self.llm.invoke(prompt).content
        return response.strip()
