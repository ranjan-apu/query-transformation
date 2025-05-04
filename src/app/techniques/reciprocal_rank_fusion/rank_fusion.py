from typing import List, Dict

from langchain_core.documents import Document

from src.app.techniques.base import BaseRAGTechniques


class ReciprocalRankFusion(BaseRAGTechniques):

    def retrieve(self, query: str,k: int = 5) -> List[Document]:
        semantic_results = self.db.similarity_search(query,k=k*2)
        
        expanded_query = self._expand_query(query)
        keyword_results = self.db.similarity_search(expanded_query, k = k*2)

        fused_results = self._reciprocal_rank_fusion([semantic_results,keyword_results])

        return fused_results[:k]
        

    def _expand_query(self, query: str)->str:
        prompt = f"""Generate an expanded version of this query for the query to implement Reciprocal Rank Fusion. Add Relevant Keywords.
        Query: {query} Expanded Query:"""
    
        response = self.llm.invoke(prompt).content
        return response.strip()


    def _reciprocal_rank_fusion(self, result_lists:List[List[Document]], k: int = 60)->List[Document]:
        doc_scores: Dict[str,float] = {}
        unique_docs:Dict[str,Document] = {}

        for results in result_lists:
            for rank,doc in enumerate(results):
                doc_id = hash(doc.page_content)
                doc_id_str = str(doc_id)

                unique_docs[doc_id_str] = doc

                rrf_score = 1.0/(rank+k)
                doc_scores[doc_id_str] = doc_scores.get(doc_id_str, 0)+rrf_score


        sorted_doc_ids = sorted(doc_scores.keys(),key=lambda x:doc_scores[x],reverse=True)

        return [unique_docs[doc_id] for doc_id in sorted_doc_ids]



        
