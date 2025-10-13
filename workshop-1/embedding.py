from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment")

# Initialize Gemini embeddings (LangChain integration)
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=google_api_key,
)

# Documents list (example)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
]

# Natural-language query
query = "tell me about captain with great finishing skills"

# Compute embeddings
doc_embeddings = embedding.embed_documents(documents) 
query_embedding = embedding.embed_query(query)       
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Pick best match
index = np.argmax(scores)
score = scores[index]

print("*" * 40)
print(scores)
print("*" * 40)
print(index, score)
print("*" * 40)
print(query)
print(documents[index])
