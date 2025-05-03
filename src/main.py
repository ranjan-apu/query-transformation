# test_parallel_query.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rich.console import Console

from src.app.techniques.parallel_query import ParallelQueryFanOut
from src.db.qdrant_client import QdrantDB
from src.utils.pdf_loader import load_pdf
from src.utils.text_splitter import split_documents


def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    console = Console()
    console.print("[bold green]Testing Parallel Query Fan Out RAG Technique[/bold green]")

    # Ask for PDF path
    pdf_path = input("Enter the path to your PDF file: ")

    if not os.path.exists(pdf_path):
        console.print(f"[red]Error: File {pdf_path} does not exist[/red]")
        return

    try:
        # Initialize components
        embeddings = OpenAIEmbeddings(api_key=api_key)
        llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14",api_key=api_key, temperature=0)
        db = QdrantDB()

        # Load and process PDF
        console.print(f"[green]Loading PDF from {pdf_path}...[/green]")
        documents = load_pdf(pdf_path)
        console.print(f"[green]Loaded {len(documents)} pages.[/green]")

        console.print("[green]Splitting documents...[/green]")
        chunks = split_documents(documents)
        console.print(f"[green]Split into {len(chunks)} chunks.[/green]")

        # Add to vector database
        console.print("[green]Adding to vector database...[/green]")
        db.create_collection(embeddings)
        db.add_documents(chunks)
        console.print("[green]Documents successfully added to the database![/green]")

        # Initialize the RAG technique
        parallel_rag = ParallelQueryFanOut(db, embeddings, llm)

        # test_parallel_query.py (updated snippet)
        # Interactive query loop
        console.print("[green]Ready for queries! Type 'exit' to quit.[/green]")
        while True:
            query = input("\nYour question: ")
            
            if query.lower() == "exit":
                break
            
            # Show the alternative queries generated
            alt_queries = parallel_rag._generate_alternative_queries(query)
            console.print("[bold cyan]Generated Alternative Queries:[/bold cyan]")
            for i, q in enumerate(alt_queries):
                console.print(f"[cyan]Query {i+1}: {q}[/cyan]")
            
            # Get the answer using the parallel query technique
            console.print("[green]Retrieving documents using all queries and generating answer...[/green]")
            
            # Explicitly show retrieved documents for each query (for testing clarity)
            all_docs = []
            doc_ids = set()
            for alt_query in alt_queries:
                docs = parallel_rag.db.similarity_search(alt_query, k=3)
                console.print(f"[yellow]Documents retrieved for '{alt_query}': {len(docs)}[/yellow]")
                for doc in docs:
                    doc_id = hash(doc.page_content)
                    if doc_id not in doc_ids:
                        all_docs.append(doc)
                        doc_ids.add(doc_id)
            
            console.print(f"[yellow]Total unique documents retrieved: {len(all_docs)}[/yellow]")
            
            # Generate answer using the retrieved documents
            answer = parallel_rag.generate(query, all_docs)
            
            console.print("[bold blue]Answer:[/bold blue]")
            console.print(f"[blue]{answer}[/blue]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    main()
