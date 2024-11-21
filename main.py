import click
from rich.console import Console
import os
import glob
import docx
import fitz
import spacy
import uuid
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client


console = Console()

db_directory = "./db"
client = chromadb.PersistentClient(path=db_directory)
collection = client.get_or_create_collection(name="my_collection")
embedding_model = embedding_functions.DefaultEmbeddingFunction()

@click.group()
def cli():
    """second brain - ask anything about your documents""" 
    pass

@cli.command()
@click.argument('directory')
def init(directory):
    if not os.path.isdir(directory):
        console.print(f"[bold red]Error:[/] The specified directory {directory} does not exist.")
        return
    
    console.print(f'Initializing RAG pipeline from {directory}...')

    all_files = glob.glob(os.path.join(directory, "*"))
    for file in all_files:
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext == ".pdf":
            document_text = extract_text_from_pdf(file)
        elif file_ext == ".docx":
            document_text = extract_text_from_word(file)
        elif file_ext == ".txt":
            document_text = extract_text_from_txt(file)
        else:
            continue  # Skip unsupported file types
    
        document_chunks = split_text_into_chunks(document_text)
        for chunk in document_chunks:
            embeddings = embedding_model([chunk])
            unique_id = str(uuid.uuid4()) 
            collection.add(
                embeddings=embeddings,
                ids=[unique_id],
                documents=[chunk],
                metadatas=[{"filename": os.path.basename(file)}],
            )
            console.print(embeddings)
        console.print(f"Indexed chunk from {file}")
    

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def extract_text_from_word(word_path):
    """Extract text from a Word document"""
    doc = docx.Document(word_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file"""
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def split_text_into_chunks(text, chunk_size=1000):
    """Split the text into smaller chunks to create more manageable embeddings"""
    """Split by sentences"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sentence in doc.sents:
        if len(current_chunk) + len(sentence.text) <= chunk_size:
            current_chunk += sentence.text
        else:
            chunks.append(current_chunk)
            current_chunk = sentence.text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

@cli.command()
def get():
    """Retrieve all documents, embeddings, and metadata from the collection"""
    stored_data = collection.get(include=['embeddings', 'documents', 'metadatas'])
    for idx, document in enumerate(stored_data["documents"]):
        console.print(f"ID: {stored_data['ids'][idx]}")
        console.print(f"Document: {document}")
        console.print(f"Metadata: {stored_data['metadatas'][idx]}")
        console.print(f"Embedding: {stored_data['embeddings'][idx][:5]}...")  # Check the embeddings
        console.print("-" * 50)


def generate_response_with_llm(prompt):
    ollama_client = Client(host='http://localhost:11434')
    response = ollama_client.chat(model='llama3.2:1b', messages=[
        {
            'role': 'user',
            'content': f'{prompt}',
        },
    ])
    return response['message']['content']



@cli.command()
@click.argument('query')
def ask(query):
     # Retrieve the relevant documents using the query
    query_embedding = embedding_model([query])
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    console.print(f"Results: {results['documents']}")
    # Combine retrieved results and generate response
    context = "\n".join(results['documents'][0])
    prompt = f"Answer the following question based on the context:\n{context}\nQuestion: {query}"
    
    # Generate the response using an LLM
    response = generate_response_with_llm(prompt)
    console.print(f"Response: {response}")

if __name__ == '__main__':
    cli()