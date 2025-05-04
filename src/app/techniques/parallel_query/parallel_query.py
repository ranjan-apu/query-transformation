from typing import List

from langchain_core.documents import Document

from src.app.techniques.base import BaseRAGTechniques


class ParallelQueryFanOut(BaseRAGTechniques):

    def retrieve(self, query: str) -> List[Document]:
        alternative_queries = self._generate_alternative_queries(query)

        all_docs = []
        doc_ids = set()

        for alt_query in alternative_queries:
            docs = self.db.similarity_search(alt_query)

            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in doc_ids:
                    all_docs.append(doc)
                    doc_ids.add(doc_id)

        return all_docs

    def _generate_alternative_queries(self, query: str) -> List[str]:
        prompt = f"""Generate minimum of 3 alternative search queries that capture different aspects of the original query.Also Alternate Query should capture possible spelling mistake and if the use query is in the different language we should have comprehnsive query in the english
        Format your response exactly as follows:
        Query 1: [first alternative query]
        Query 2: [second alternative query]
        Query 3: [third alternative query]

        Original query: {query}"""

        response = self.llm.invoke(prompt).content

        queries = [query]
        for line in response.strip().split('\n'):
            if line.startswith('Query '):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    clean_query = parts[1].strip()
                    if clean_query and clean_query not in queries:
                        queries.append(clean_query)

        return queries[:4]

