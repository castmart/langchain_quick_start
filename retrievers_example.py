from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

## LLM
llm = ollama.Ollama(model='llama2')

# Retrievers
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

embeddings = OllamaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Document Chain creation
prompt = ChatPromptTemplate.from_template(""" Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

while True:
    print("\n\n----")
    print("\n\n----")
    user_input = input("prompt@here$ ")
    print("\n\n----")
    response = retriever_chain.invoke({"input": user_input})
    print(response["answer"])
