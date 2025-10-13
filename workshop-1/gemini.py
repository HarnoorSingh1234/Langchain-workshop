from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

hehe = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_output_tokens=5000)
result = hehe.invoke("Write a poem about a lonely computer.")
print(result.content)
