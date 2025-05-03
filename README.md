# RAG-PDF-CLI: Advanced Retrieval-Augmented Generation for PDFs

Welcome to **RAG-PDF-CLI**, a Python-based command-line interface (CLI) application designed to demonstrate advanced Retrieval-Augmented Generation (RAG) techniques for processing and querying PDF documents. This project is built for learning purposes, showcasing seven cutting-edge RAG methods using LangChain for orchestration and Qdrant as a vector database running in Docker.

Whether you're a developer, researcher, or AI enthusiast in Bengaluru or beyond, this project offers a hands-on way to explore how advanced RAG can enhance information retrieval and generation from unstructured text like PDFs.

## Table of Contents
- [Overview](#overview)
- [Advanced RAG Techniques](#advanced-rag-techniques)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Comparison of Techniques](#comparison-of-techniques)
- [Contributing](#contributing)
- [License](#license)

## Overview
RAG combines retrieval (finding relevant documents) with generation (crafting coherent answers) to provide accurate, context-aware responses. This project goes beyond basic RAG by implementing advanced techniques that tackle challenges like query ambiguity, retrieval coverage, and answer quality. It processes a PDF, stores embeddings in Qdrant, and lets you query the content using different RAG strategies via a CLI.

The goal? To help you understand and experiment with state-of-the-art methods in a modular, easy-to-follow codebase.

## Advanced RAG Techniques
This project implements seven advanced RAG techniques, split into retrieval and generation categories:

### Retrieval Techniques
1. **Query Transformation**: Rewrites user queries into formal language to better match document phrasing, improving retrieval relevance.
2. **Parallel Query Fan-Out Retrieval**: Generates multiple paraphrases of the query and retrieves documents for each in parallel, ensuring broader coverage.
3. **Reciprocal Rank Fusion (RRF)**: Merges results from multiple retrieval methods or query variants by assigning scores based on rank, creating a unified, high-quality result set.
4. **Query Decomposition**: Breaks complex queries into simpler sub-questions, retrieving documents for each to handle multifaceted inquiries.

### Generation Techniques
5. **Step-Back Prompting**: Encourages the model to critique and refine its initial answer, improving accuracy through self-reflection.
6. **Few-Shot Prompting**: Provides example question-answer pairs in the prompt to guide the model toward better response formatting and content.
7. **Hypothetical Document Embedding (HyDE)**: Creates a "virtual" document summary from the query and uses its embedding to retrieve relevant real documents, enriching context.

## Project Structure
The codebase is modular, with each technique isolated in its own file for clarity and experimentation.

```
rag-pdf-cli/
├── docker-compose.yml          # Docker configuration for Qdrant vector DB
├── requirements.txt            # Python dependencies
└── app/
    ├── ingestion.py            # PDF loading, chunking, and embedding storage
    ├── cli.py                 # Main CLI to interact with RAG techniques
    ├── retrieval/             # Retrieval-focused RAG techniques
    │   ├── query_transformation.py
    │   ├── parallel_fanout.py
    │   ├── reciprocal_rank_fusion.py
    │   └── query_decomposition.py
    └── generation/            # Generation-focused RAG techniques
        ├── step_back_prompting.py
        ├── few_shot_prompting.py
        └── hypothetical_embedding.py
```

## Prerequisites
Before setting up the project, ensure you have:
- **Docker** and **Docker Compose** installed to run Qdrant.
- **Python 3.8+** for running the application.
- A PDF file to ingest and query.
- An API key for OpenAI if using LLM-based techniques (optional, replaceable with local models).

## Setup Instructions
Follow these steps to get the project up and running:

1. **Clone the Repository**  
   Clone or download this project to your local machine.

   ```bash
   git clone 
   cd rag-pdf-cli
   ```

2. **Start Qdrant with Docker**  
   Qdrant is used as the vector database to store embeddings. Launch it using Docker Compose.

   ```bash
   docker-compose up -d
   ```

   This starts Qdrant on `http://localhost:6333`. Verify it's running by checking the logs or visiting the endpoint in a browser (if the UI is enabled).

3. **Install Python Dependencies**  
   Create a virtual environment (optional but recommended) and install the required packages.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Prepare a PDF**  
   Have a PDF file ready to ingest. This will be chunked and stored as embeddings in Qdrant.

## Usage
The CLI provides two main commands: `load` to ingest a PDF and `ask` to query it using a specific RAG technique.

1. **Ingest a PDF**  
   Load your PDF into the system. This extracts text, chunks it, generates embeddings, and stores them in Qdrant.

   ```bash
   python app/cli.py load path/to/your/document.pdf
   ```

   Example output:
   ```
   Ingested 42 chunks into pdf_rag
   ```

2. **Query the PDF**  
   Ask a question and specify a RAG technique using the `--method` flag. Available methods are: `transform`, `fanout`, `rrf`, `decompose`, `stepback`, `fewshot`, and `hypo`.

   ```bash
   python app/cli.py ask "What is the main topic of the document?" --method fanout
   ```

   Example output:
   ```
   Retrieved content using Parallel Query Fan-Out:
   [Relevant text chunks from the PDF...]
   ```

   Try different methods for the same query to compare results!

## Comparison of Techniques
Each RAG technique addresses specific challenges in retrieval or generation. Here's a quick comparison:

| **Technique**               | **Strength**                                      | **Best Use Case**                       |
|-----------------------------|--------------------------------------------------|----------------------------------------|
| Query Transformation        | Aligns query wording to document style           | When queries are casual or vague       |
| Parallel Query Fan-Out      | Broadens retrieval with query variants           | When coverage of topics is critical    |
| Reciprocal Rank Fusion      | Combines multiple rankings for better results    | When using multiple retrieval sources  |
| Query Decomposition         | Handles complex queries via sub-questions        | For multi-part or intricate questions  |
| Step-Back Prompting         | Improves answer quality through self-critique    | When initial answers seem incomplete   |
| Few-Shot Prompting          | Guides model with examples for consistency       | When specific answer format is needed |
| Hypothetical Embedding      | Enriches context with virtual summaries          | When query lacks direct matches        |

Experiment with each method to see how they impact retrieval relevance and answer quality for your specific PDF content.

## Contributing
This is a learning project, and contributions are welcome! If you have ideas for additional RAG techniques, optimizations, or bug fixes, feel free to:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a clear description of your changes.

Please ensure your code follows PEP 8 style guidelines and includes comments for clarity.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it for educational or personal purposes.

---

Happy learning and experimenting with advanced RAG techniques! If you have questions or run into issues, don't hesitate to reach out or open an issue. Let's build smarter retrieval systems together! 🚀

---