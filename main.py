# to start - streamlit run main.py
# http://localhost:8501/


import streamlit as st
from openai import OpenAI

st.title("My first ChatBot")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=st.secrets["GITHUB_TOKEN"]
    )

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
            "role": "assistant",
            "content": "Always respond in lithuanian language, even if asked in any other language"
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

    ##### add instruction/prompt enginering to get the response from the model
    ##### add file aploading to get the response from the model
    ##### integrate RAG, langchain

    