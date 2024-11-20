import click
from rich.console import Console
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

console = Console()
client = chromadb.Client()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

collection = client.create_collection(name="my_documents")

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
    # Split by paragraphs or sentences or any other method
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

@cli.command()
@click.argument('prompt')
def ask(prompt):
    console.print(f'Prompt: {prompt}')

if __name__ == '__main__':
    cli()