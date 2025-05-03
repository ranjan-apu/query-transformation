from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        return 'Error: File not found.'
    except Exception as e:
        return f"Error: {str(e)}"