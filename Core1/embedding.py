import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() # start of function to safely retrieve API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # gives OpenAI access to API key but is accessed safely from a file so no one else can read

sentences = [
    "The TV fell off the wall.",
    "The TV crashed to the ground", # basic version of what should be the content of the PDF file
    "A water bottle was filled."
]

embeddings = []

for sentence in sentences: # loops through list of sentences so a single sentence can be used as input
    response = client.embeddings.create(
        input=sentence,
        model="text-embedding-3-small"
    ) # response becomes output after the sentence gets its embedded values
    embedding = response.data[0].embedding # this extracts the embedded values from the output
    embeddings.append(embedding) # adds the values to the list
    print(f"Sentence : {sentence}") # displays sentence
    print(f"Embedding : [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}], ...") #displays first 3 values of embedding to 4 dp
    print(f"Length : {len(embedding)} numbers\n") # prints how many value are in the embedding


def dot_product(a,b):
    return sum(x * y for x, y in zip(a,b)) # formula for dot product (see documentation for more detail)

sim1_2 = dot_product(embeddings[0], embeddings[1])
sim1_3 = dot_product(embeddings[0], embeddings[2]) # these lines find the dot product of the embeddings of sentences 1&2 , 1&3

print(f"-" * 50) # presentation
print(f"Similarity (1&2) : {sim1_2:.4f} (similar meaning)") # similar sentences should score higher
print(f"Similarity (1&3) : {sim1_3:.4f} (different meaning)") # different sentences should score lower


  

 