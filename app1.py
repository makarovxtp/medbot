from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage


import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

embeddings = download_hugging_face_embeddings()

# Initializing Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "medical-chatbot"

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Mixtral-8x7b-32768")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)




retriever = docsearch.as_retriever(search_kwargs={'k': 2})

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    global chat_history

    if request.method == "POST":
        msg = request.form["msg"]
        input = msg
        print(input)

        # Use the conversational RAG chain to handle the input
        result = rag_chain.invoke({"input": input, "chat_history": chat_history})
        
        # Update chat history
        chat_history.append(HumanMessage(content=input))
        chat_history.append(AIMessage(content=result["answer"]))
        
        print("Response:", result["answer"])
        return str(result["answer"])
    else:
        return jsonify({"error": "Invalid request method"})

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=8080, debug=True)
    app.run()
