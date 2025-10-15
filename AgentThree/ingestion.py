from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilanweng.github.io/posts/2023-06-23-agent/",
    "https://lilanweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilanweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=0,
)
docs_split = text_splitter.split_documents(docs_list)

# Comment when database is created
vectorstore = Chroma.from_documents(
    documents=docs_split,
    collection_name="rag-chrome",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma_db",
)

# Uncomment to use retriever
# retriever = Chroma(
#     collection_name="rag-chrome",
#     embedding_function=OpenAIEmbeddings(),
#     persist_directory="./.chroma_db",
# ).as_retriever()

if __name__ == "__main__":
    print("Hello LangGraph!")
    # print(retriever.get())