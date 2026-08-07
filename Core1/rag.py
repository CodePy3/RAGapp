import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv() # start of function to safely retrieve API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # gives OpenAI access to API key but is accessed safely from a file so no one else can read

documents = [
    "Items can be returned within 30 days of purchase for a full refund.",
    "Products must be unused and in their original packaging to be eligible for return.",
    "Refunds are issued to the original payment method within 5-7 business days.",
    "Sale items are final and cannot be returned or exchanged.",
    "Customers are responsible for return shipping costs unless the item was defective."
]

chroma_client = chromadb.Client() # alias 
collection = chroma_client.create_collection(name="myFirstRag") # alias for collection

print(f"Embedding and storing documents...")

for i,doc in enumerate(documents): # enumerate used to create pairs (i, doc)
    embedding = client.embeddings.create(
        input=doc, # turns doc into embedded values
        model="text-embedding-3-small"
    ).data[0].embedding # extracts embedded values cleanly

    collection.add(
        documents=[doc], # stores it in text form
        embeddings=[embedding], # stores it in vector/number form
        ids=[f"doc_{i}"] # number from pair is used as ID
    )

print(f"stored {len(documents)} documents in ChromaDB\n")

###### QUESTIONS #####
question = "I have had my item for 10 days it is unused, but it was on sale item, can i return it?"
                
question_embedding = client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
).data[0].embedding # extracts embedded values cleanly

results = collection.query( #queries ChromaDB entries
    query_embeddings=[question_embedding], # looks for entries nearest to questions embedded values
    n_results=3 # returns the 2 closest embedded values in the DB
)

retrieved_chunks = results["documents"][0] # stores a list of all the chunks retrieved from the query 
print("Chunks retrieved:")
for chunk in retrieved_chunks: # print them out
    print(f"{chunk}")

context = "\n".join(retrieved_chunks) # this allows any chunks in the variable to be cleanly added into the prompt

prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:""" # creates prompt to give to API

print(prompt)


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content
print(f"Question: {question}")
print(f"Answer: {answer}")