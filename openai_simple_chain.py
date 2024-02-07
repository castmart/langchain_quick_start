from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Model
llm = ChatOpenAI()  # API KEY should be set as Env variable

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class Web Designer"),
    ("user", "{input}")
])

# Output Parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

chain_response = chain.invoke({"input": "how can Jimdo help me with selling my products online?"})

print(chain_response)
