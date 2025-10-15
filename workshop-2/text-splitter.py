from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("workshop-2/notes.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="")
# chunks = splitter.split_documents(docs)

result = splitter.split_documents(docs)

print(result[1].page_content)