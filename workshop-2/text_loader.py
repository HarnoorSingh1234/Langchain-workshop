from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


prompt = PromptTemplate(
    template='write a summarised part of the person described in the text: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader("workshop-2/spy.txt", encoding="utf-8")

docs = loader.load()

result = llm.invoke(docs[0].page_content)

print(result.content)