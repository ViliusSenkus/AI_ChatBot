# to start - streamlit run main.py
# http://localhost:8501/


import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import io

st.title("My first ChatBot")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=st.secrets["GITHUB_TOKEN"]
    )

# File uploaders
uploaded_picture = st.file_uploader("Choose picture to discuss on", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if uploaded_picture is not None:
    st.image(uploaded_picture, caption="Jūsų įkelta nuotrauka diskusijai")

uploaded_document = st.file_uploader("Choose document to analyze", type=["pdf", "txt", "doc", "docx"], accept_multiple_files=False)

merged_documents = []
documents_names = []
chunk_size = 1000
chunk_overlap = 200

def read_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())  # Įrašome įkeltą failą į laikiną failą
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def read_txt(file):
    text = file.getvalue()
    docs=[]
    i = 0
    while i < len(text):
        if i + chunk_size > len(text):
            chunk = text[i:]
        else:
            chunk = text[i:i + chunk_size]
        docs.append(chunk)
        i += chunk_size - chunk_overlap
    return docs
    
######!!!!!!!!!!!!!!!!!!!!!!!!!!!REIKIA SUDETI KELIS DOKUMNETUS

def load_document(document):
    if document is not None:
        if document.type == "application/pdf":
            text = read_pdf(document)
            st.write(text)
        elif document.type == "text/plain":
            text = read_txt(document)
            st.write(text)
        else:
            st.error("MS word files are not supported yet.")

load_document(uploaded_document)

#vectorising data

# initiate chat session to keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    full_response = client.chat.completions.create(
    messages=[

        {
            #  instruction/prompt enginering example - 1
            "role": "system",
            "content": "Always respond in lithuanian language, even if asked in any other language"
        },
        {
            #  instruction/prompt enginering example - 2
            "role": "system",
            "content": "pretend as you are an famoust traveler. Be polite, helpful, informal. Answer with no more than 3 short sentences. Allways add travel tips"
        },
        {
            #  instruction/prompt enginering example - 3
            "role": "assistant",
            "content": "rephrase user question in the begining of answer and ask related question on answer content at the end of the answer"
        },
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-4o",
    # temperature=1,
    max_tokens=4096,
    # top_p=1
)

    response = full_response.choices[0].message.content
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


    ##### add file aploading to get the response from the model
    ##### integrate RAG, langchain

    