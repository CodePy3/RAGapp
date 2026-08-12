import os 
import pymupdf as fitz
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import chromadb

load_dotenv() # start of function to safely retrieve API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # gives OpenAI access to API key but is accessed safely from a file so no one else can read

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./chroma_db") # creates a ChromaDB with permanent storage (key: Persistent)

def get_or_create_collection():
     try:
          return chroma_client.get_collection("rag_collection")
     except:
          return chroma_client.create_collection("rag_collection")

chunk_size = 500
chunk_overlap = 50

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i,page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append((i+1, text))
    doc.close()
    return pages 

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    chunks = [] # create list for chunks
    start = 0 # start at first word in text
    while start < len(text): # loops through text until end is reached
        end = start + chunk_size 
        chunk = text[start:end] # slices text to split text into chunks
        chunks.append(chunk) 
        start += chunk_size - chunk_overlap # next chunk shares 50 words with previous to solve the split problem
    return chunks   

def embed_texts(texts):
    response = client.embeddings.create(
        input=texts, # converting text to embedded values 
        model="text-embedding-3-small"
    ) 
    return [item.embedding for item in response.data] # pulls out just the vector for each text in response

@app.post("/ingest")
async def ingest(file:UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    collection = get_or_create_collection()

    pages = extract_text(tmp_path)
    all_chunks = []
    all_metadata = [] 
    all_ids = []
    chunk_index = 0
    
    for page_num, page_text in pages: # looping through text from pdf and placing text and page nums into pairs
        chunks = chunk_text(page_text) # splitting pdf text into chunks with previous function

        for chunk in chunks:
            if len(chunk.strip()) < 50: # if after whitespace is removed the chunk is less than 50 words it is removed
                continue
            all_chunks.append(chunk) # puts chunks into a list
            all_metadata.append({
                "source": os.path.basename(tmp_path),
                "page": page_num,
                "chunk": chunk_index,
            }) # stores meta data about each chunk

            all_ids.append(f"chunk_{chunk_index}") # creates list of chunk id's
            chunk_index += 1

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(all_chunks),batch_size): # increments by 100 (batch size)
        batch  = all_chunks[i : i + batch_size] # 100 chunks are placed into batch variable
        embeddings = embed_texts(batch) # embeds the batch using function from before
        all_embeddings.extend(embeddings) # adds it to the list of embedded values

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids
    )

    os.unlink(tmp_path)

    return {
        "message": f"Ingested {file.filename} successfully",
        "chunks_added": len(all_chunks),
        "total_chunks": collection.count()
    }

class AskRequest(BaseModel): # BaseModel creates a expectation of what data types you want to recieve
    question:str
    n_results:int=3 # now any AskRequest must contain a string and an int

@app.post("/ask")
async def ask(request:AskRequest):
    collection = get_or_create_collection()

    if collection.count() == 0:
        return {"error":"No documents ingested yet"}

    question_embedding = embed_texts([request.question])[0]

    results = collection.query( #queries ChromaDB entries
        query_embeddings=[question_embedding], # looks for entries nearest to questions embedded values
        n_results= request.n_results, # returns the n closest embedded values in the DB
        include = ["documents","metadatas","distances"] # returns all other details from collection

    )

    chunks = results["documents"][0] # gives variable name to list of texts (chunks) stored in collection
    metadatas = results["metadatas"][0] # gives variable name to metadata list 
    distances = results["distances"] [0] # variable name for distance (how disimilar a chunk is to question)

    context_parts = []
    sources = []

    for chunk, meta, dist in zip(chunks, metadatas, distances):
        context_parts.append(chunk)
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "score": round(dist,4),
        })

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the relevant context below.
Be concise and specific.
Always mention the source document and page if relevant.
If the answer is not in the context, say "I don't know based on the provided documents."
    
Context:
{context}

Question: {request.question}
Answer: """ # creates prompt

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        ) # typical way to send message to model and recieve reply 

    return {
        "question": request.question,
        "answer": response.choices[0].message.content,
        "sources": sources,
    }
    
@app.get("/healthCheck")
def root(): 
    collection = get_or_create_collection()
    return {
        "status": "running",
        "total_chunks": collection.count(),
    } # this code section is a health check for the app