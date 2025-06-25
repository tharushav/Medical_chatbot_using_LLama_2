from flask  import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

# Load the index
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

llm=CTransformers(model=model_path,
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
            
        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"error": "Empty message"}), 400
            
        print(f"User input: {msg}")
        
        # Get response from QA chain
        result = qa({"query": msg})
        
        if not result or "result" not in result:
            return jsonify({"error": "Failed to generate response"}), 500
            
        response_text = result["result"].strip()
        if not response_text:
            response_text = "I'm sorry, I couldn't find relevant information to answer your question."
            
        print(f"Response: {response_text}")
        return jsonify({"reply": response_text})
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get", methods=["GET", "POST"])
def get_chat():
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return "Please provide a message"
            
        print(f"User input: {msg}")
        result = qa({"query": msg})
        response_text = result.get("result", "").strip()
        
        if not response_text:
            response_text = "I'm sorry, I couldn't find relevant information to answer your question."
            
        print(f"Response: {response_text}")
        return response_text
        
    except Exception as e:
        print(f"Error in get_chat endpoint: {str(e)}")
        return "Sorry, there was an error processing your request."

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug= True)