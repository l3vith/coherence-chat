import os
import getpass
from groq import Groq
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain import hub
from langgraph.graph import START, StateGraph
from pydantic.main import BaseModel
from typing_extensions import List, TypedDict

import re
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

'''
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
'''    

load_dotenv()

print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')}")
print(f"HUGGING_FACE_API_KEY: {os.getenv('HUGGING_FACE_API_KEY')}")


llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq", api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = os.getenv('HUGGING_FACE_API_KEY'), 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = InMemoryVectorStore(embedding=embeddings)

# Data - 1 and Data - 2
data_1 = open(r'data_1.txt', 'r').read()
data_2 = open(r'data_2.txt', 'r').read()
data_3 = open(r'data_3.txt', 'r').read()
data_4 = open(r'data_4.txt', 'r').read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_text(data_1 + "\n\n" + data_2 + "\n\n" + data_3 + "\n\n" + data_4)
docs = [Document(page_content=text) for text in all_splits]
_ = vector_store.add_documents(documents=docs)

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
'''
response = graph.invoke({"question": "Who should i contact for help ?"})
print(response["answer"])
'''

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Pong!"
    
class Query(BaseModel):
    question: str

@app.get("/chat")
async def chat(request: Query):
    response = graph.invoke({"question": request.question})
    response = response["answer"]
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL) 
    return response
