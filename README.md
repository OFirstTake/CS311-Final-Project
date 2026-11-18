# CS Major Handbook RAG Chatbot

## Description

A RAG chatbot that answers questions about the Computer Science program based on `CS_MajorHandbook.pdf`. The application uses Streamlit for the user interface, Chroma to retrieve relevant content, and Ollama to generate embeddings and answers.

## Features

- Extracts and indexes PDF content by page.
- Retrieves document sections relevant to each question.
- Generates answers using a language model served by Ollama.
- Displays PDF sources and page numbers.
- Saves conversation history and Q&A results in the data directory.

## Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
RAG pipeline (LangChain)
  |                         |
  v                         v
Chroma vector store       Ollama
(PDF content)             Embeddings + LLM
```

The application retrieves up to five relevant sections from Chroma and sends them, along with the question, to Ollama. If the documents are not sufficiently relevant, the chatbot indicates that it is using the model's general knowledge.

## Docker Requirements

- Docker Desktop or Docker Engine with Docker Compose.
- Ollama installed and running on the host machine.
- The required Ollama models downloaded. Ollama uses port `11434` by default.

## Run Ollama and Docker

### 1. Download the Ollama models

Install Ollama from <https://ollama.com/>, then run these commands on the host machine:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

Verify that the models are available:

```bash
ollama list
```

### 2. Build and start the application (first run)

From the repository root, build the image and start the container:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Open the application at <http://localhost:8501>. Docker Compose configures the app to connect to Ollama on the host at `host.docker.internal:11434` and mounts the data directory for conversation history and the vector store.

### 3. Start the application again

After the image has been built, start the existing container without rebuilding:

```bash
docker compose -f docker/docker-compose.yml up
```

Use `--build` again if you change the Dockerfile, dependencies, or application code and want those changes included in a new image.

Stop the application with:

```bash
docker compose -f docker/docker-compose.yml down
```

## Usage Example

Enter a question in the chat box, for example:

```text
What are the requirements for the Computer Science major?
```

Other example questions:

- What courses are required to formally declare the Computer Science major?
- What are the learning outcomes for the Computer Science major?
- What career paths can a Computer Science graduate pursue?

When relevant information is found in the handbook, the application displays the answer along with its source sections and page numbers.
