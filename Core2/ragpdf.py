import os
from dotenv import load_dotenv
from openai import OpenAI
import pymupdf as fitz
import chromadb

load_dotenv() # start of function to safely retrieve API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # gives OpenAI access to API key but is accessed safely from a file so no one else can read

pdf_path="TestPDF.pdf"
chunk_size=500
chunk_overlap=50


def extract_text_pdf(pdf_path):
    doc = fitz.open(pdf_path) # allows fitz to access pdf document i want to read
    pages = [] # list to store pages
    for page_num, page in enumerate(doc): # loop through pages in document, use enumerate to save page number in pair
        text = page.get_text() # get text from pdf
        text = text.strip() # removes leading and trailing whitespace
        if text:
            pages.append((page_num + 1,text)) # adds page to the list in the correct index
    doc.close() # close document after its finished
    print(f"Extracted text from {len(pages)} pages") # tells you how many pages there are
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
    

def  index_pdf(pdf_path):
    chroma_client = chromadb.PersistentClient(path="./chroma_db") # creates a ChromaDB with permanent storage (key: Persistent)

    try:
        chroma_client.delete_collection("pdf_rag") # checks if collection already exists and deletes it if it does
    except:
        pass 

    collection = chroma_client.create_collection("pdf_rag")

    pages = extract_text_pdf(pdf_path)

    # initialising variables
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
                "source": os.path.basename(pdf_path),
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
        print(f"Embedded batch {i // batch_size+1} ({len(batch)} chunks)") # tell you which batch has been embedded

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids
    ) # adds all data we just created to the correct categories in collection
    print(f"\nDone! Indexed {len(all_chunks)} chunks from {pdf_path}")
    return collection

def ask(collection, question, n_results):
    question_embedding = embed_texts([question])[0] 

    results = collection.query( #queries ChromaDB entries
    query_embeddings=[question_embedding], # looks for entries nearest to questions embedded values
    n_results= n_results, # returns the 2 closest embedded values in the DB
    include = ["documents","metadatas","distances"] # returns all other details from collection

)

    chunks = results["documents"][0] # gives variable name to list of texts (chunks) stored in collection
    metadatas = results["metadatas"][0] # gives variable name to metadata list 
    distances = results["distances"] [0] # variable name for distance (how disimilar a chunk is to question)

    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks:")
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        print(f"[Page {meta['page']} | score: {dist:.3f}] {chunk[:100]}") # print out query in neat way (e.g. only 100 characters from each chunk)

    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Answer the question using only the context below.
Always mention which page the answer came from. 
If the answer is not in the context say, "I don't know."
    
Context:
{context}

Question: {question}
Answer: """ # creates prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ) # typical way to send message to model and recieve reply 

    print(f"\nAnswer: {response.choices[0].message.content}") # print out models answer

if __name__ == "__main__":
    collection = index_pdf(pdf_path)
    ask(collection,"What is the hyrdological cycle?", 3)
    ask(collection,"What is Marlowe Bay? ", 3)