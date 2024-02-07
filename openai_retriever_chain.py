from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Model
llm = ChatOpenAI()  # API KEY should be set as Env variable

# Document load
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Create a vectorstore (a.k.a Embeddings)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)  # Vector store (indexed and ingested by OpenAIEmbeddings)

# Prompt
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retriever and retrieval chain to explicitly load the context (loaded documents)
# instead of doing it manually.
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

chain_response = retrieval_chain.invoke({"input": "how can langsmith help me knowing how many tokens have been spent?"})

print(chain_response["answer"])
