from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


prompt = PromptTemplate(
    template='breifly tell about the product in the link and its price only: {text} do it like a 5 line poem',
    input_variables=['text']
)

parser = StrOutputParser()

loader = WebBaseLoader("https://www.amazon.in/Apple-iPad-Air-11%E2%80%B3-chip/dp/B0DZ78JYYK?ref=dlx_prime_dg_dcl_B0DZ78JYYK_dt_sl8_03_pi&pf_rd_r=EAGQW7C4C51CPG2HX0S1&pf_rd_p=bf0d4214-f7c3-4440-bb31-c5ecb7db1503")

docs = loader.load()
print(llm.invoke(prompt.format(text=docs[0].page_content)).content)
# print(docs[0].page_content)