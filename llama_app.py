import streamlit as st
import torch
from transformers import BitsAndBytesConfig

# llama_index
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage




######################### Data Connectors #########################
def load_text_and_get_chunks(path_to_pdfs):
    documents = SimpleDirectoryReader(path_to_pdfs).load_data()
    
    return documents

######################### Models #########################
def load_llm():
    hf_token = "#####"

    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language.
- Never generate offensive or foul language.
- Do not write "The authors" in any answer
- Do not use "[]" in any answer
- Write every answer like a list of known facts without referring to anybody or any document in the third person
- Never use references in square brackets or otherwise in the output, but provide material examples if possible.
"""

    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    # load the model with quantized features
    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_use_double_quant=True,
    )

    llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    system_prompt=SYSTEM_PROMPT,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="cuda:1",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"token": hf_token, "quantization_config": quantization_config}
    )
    return llm

def load_embeddings():
        
    embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    
    return embed_model

st.set_page_config(page_title="AMGPT", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("AMGPT, powered by LlamaIndex 💬🦙")
#st.info("AMGPT ", icon="📃")

# Remove the loading message
placeholder = st.empty()
placeholder.empty()
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!!!"}
    ]



# Create a placeholder for the loading message
placeholder = st.empty()
placeholder.text("Loading the LLM, please wait...")

# get llm
llm = load_llm()



# get embeddings
embed_model = load_embeddings()

Settings.llm = llm
Settings.embed_model = embed_model

# create vector store and index
storage_context = StorageContext.from_defaults(persist_dir="######")
vector_index = load_index_from_storage(storage_context)

# Remove the loading message
placeholder.empty()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = vector_index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

                
