from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama

## Chain creation
llm = ollama.Ollama(model='llama2')

print("Generating Chat prompt template")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

while True:
    print("\n\n----")
    print("\n----")
    print("\n----")
    user_input = input("prompt@here$ ")
    response = chain.invoke({"input": user_input})
    print("\n\n----")
    print(response)
