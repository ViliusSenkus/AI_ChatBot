# to start - streamlit run main.py
# http://localhost:8501/


import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# for vectorising
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("My first ChatBot")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=st.secrets["GITHUB_TOKEN"]
    )

############################################################
######## Picture upload and analysis #######################
############################################################

uploaded_picture = st.file_uploader("Choose picture to discuss on", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if uploaded_picture is not None:
    st.image(uploaded_picture, caption="Jūsų įkelta nuotrauka diskusijai")

############################################################
######## Documents upload and analysis #####################
############################################################

uploaded_documents = st.file_uploader("Choose document to analyze", type=["pdf", "txt", "docx"], accept_multiple_files=False)

# Parameters and functions for document analysis

merged_documents = []
documents_names = []
chunk_size = 600
chunk_overlap = 100

def load_document(document):
    if document is not None:
            if document.type == "application/pdf":
                return read_pdf(document)
            elif document.type == "text/plain":
                return read_txt(document)
            elif document.type == "application/msword" or document.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return read_doc(document)
            else:
                st.error(f"This file type ({document.type}) is not supported.")
                return []

def read_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())  # Įrašome įkeltą failą į laikiną failą
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages) 
    return chunks

def read_txt(file):
    text = file.getvalue().decode('utf-8')
    chunks=[]
    i = 0
    while i < len(text):
        if i + chunk_size > len(text):
            chunk = text[i:]
        else:
            chunk = text[i:i + chunk_size]
        doc = Document(page_content=chunk, metadata={
           "source": file.name,
            "chunk_index": f"from {i} to {i + len(chunk)}"
        })    
        chunks.append(doc)
        i += chunk_size - chunk_overlap
    return chunks

def read_doc(file):
    with open("temp.docx", "wb") as f:  # Changed extension to .docx
        f.write(file.getvalue())
    loader = Docx2txtLoader("temp.docx")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    return chunks

# Processing of uploaded document

if "all_splits" not in st.session_state:
    st.session_state.all_splits = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_documents:
    splits = load_document(uploaded_documents) # List<Documents>; Documents = {page_content, metadata}
    if splits:
        st.session_state.all_splits.extend(splits)
    st.write(f"Data pieces generated: {len(st.session_state.all_splits)}")

################## Vectorising data #######################

# Example of vectorising data query based just on the uploaded documents

def create_vectorstore(documents, embedding_model_name='all-MiniLM-L6-v2'):
    if documents is not None:
        try:
            st.write(f"Creating vector store with {len(documents)} documents.")
            embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
            st.write("Embedding model loaded successfully.")
            vectorstore = FAISS.from_documents(documents, embedding_function)
            st.write("Vector store created successfully.")
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    
def query_vectorstore(vectorstore, query_text, k=5):
    try:
        if vectorstore:
            results = vectorstore.similarity_search(query_text, k=k)
            if results:
                st.write("Required data is gathered for processing", color="green")
                return results
            else:
                st.write("No data related to the query was found.", color = "red")
                return []
        else:
            print("Vector store is not initialized.", color= "red")
    except Exception as e:
        print(f"Error while searching for similar vectors: {e}")

# Create vector store if not already created
if st.session_state.all_splits and st.session_state.vectorstore is None:
    st.session_state.vectorstore = create_vectorstore(st.session_state.all_splits)
    st.write(f"Vector store created: {st.session_state.vectorstore}")

############################################################
##################### ChatBot ##############################
############################################################

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

    # Query the vector store
    prompt_content = query_vectorstore(st.session_state.vectorstore, prompt, k=5)
    if prompt_content:
        combined_content = " ".join([doc.page_content for doc in prompt_content])
    else:
        combined_content = "No relevant content found."

    # Add the combined content to the messages
    st.session_state.messages.append({"role": "system", "content": f"Search in the following content: {combined_content}"})

    full_response = client.chat.completions.create(
    messages=[
        {
            #  instruction/prompt engineering example - 1
            "role": "system",
            "content": "Always respond in Lithuanian language, even if asked in any other language"
        },
        {
            #  instruction/prompt engineering example - 2
            "role": "system",
            "content": "Pretend as you are a famous traveler. Be polite, helpful, informal. Answer with no more than 3 short sentences. Always add travel tips"
        },
        {
            #  instruction/prompt engineering example - 3
            "role": "assistant",
            "content": "Rephrase user question in the beginning of the answer and ask a related question on answer content at the end of the answer"
        },
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "system",
            "content": f"Search in the following content: {combined_content}"
        }
    ],
    model="gpt-4o",
    max_tokens=4096,
)

    response = full_response.choices[0].message.content
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})