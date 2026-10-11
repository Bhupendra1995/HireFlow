import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

print("API Key loaded:", os.getenv("GEMINI_API_KEY") is not None)

model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

vector = model.embed_query("Hello world")
print(len(vector))
print(vector[:5])