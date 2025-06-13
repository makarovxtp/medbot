🏥 Medical Chatbot with Conversational RAG using LangChain, Pinecone, and Groq
This project is an AI-powered Medical Chatbot that leverages Conversational Retrieval-Augmented Generation (RAG) to provide context-aware medical responses. It integrates Flask for the backend, Pinecone for vector storage, LangChain for chaining the retrieval and generation components, and Groq’s Mixtral-8x7b-32768 model for response generation.



🚀 Features
1.Conversational RAG pipeline for maintaining context across multiple queries.

2.Semantic search using Pinecone Vector Store.

3.Real-time chat interface with Flask.

4.Uses Hugging Face embeddings for document vectorization.

5.Context-aware retrieval with LangChain's history-aware retriever.


Powered by Groq’s LLM for fast and efficient response generation.




🛠️ Tech Stack
Flask – Web Framework

LangChain – Conversational RAG and Prompt Management

Pinecone – Vector Database for Document Search

Groq – LLM Inference API

Hugging Face – Embeddings

HTML/CSS – Frontend (Chat Interface)

dotenv – Secure API Key Management




📂 Project Structure
text
Copy
Edit
medical-chatbot/
├── app.py                # Main Flask Application
├── templates/
│   └── chat.html         # Frontend Chat UI
├── src/
│   ├── helper.py         # Utility functions (embedding download, etc.)
│   └── prompt.py         # Prompt templates for LangChain
├── .env                  # Environment Variables (API Keys)
├── requirements.txt      # Python Dependencies
└── README.md             # Project Documentation



⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory and add:

env
Copy
Edit
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
5. Run the Application
bash
Copy
Edit
python app.py
The app will start at http://127.0.0.1:5000



💬 How It Works
The user sends a message through the chat interface.

The app uses a history-aware retriever to find the most relevant documents from Pinecone.

LangChain combines the retrieved documents with the input question.

Groq’s LLM generates a context-rich answer.

Chat history is maintained to handle multi-turn conversations.




🔒 Environment Variables
PINECONE_API_KEY – Pinecone project API key

GROQ_API_KEY – Groq API key for LLM access

LANGCHAIN_API_KEY – API key to enable LangChain tracing (optional but recommended)



📌 Key Concepts
Conversational RAG: Enhances retrieval by considering the chat history for more relevant responses.

Vector Store Retrieval: Uses semantic similarity instead of keyword matching.

LangChain Chains: Modular pipelines to build complex conversational AI flows.

📸 Screenshot!

[image (2)](https://github.com/user-attachments/assets/e8c2e237-ad2b-461f-a50f-dbffab9650fa)

https://medical-chatbot-using-llm.onrender.com/



📄 License
This project is licensed under the MIT License. Feel free to use and modify it.

🙌 Acknowledgements
LangChain

Pinecone

Groq

Hugging Face

