import streamlit as st
from Functions.write_stream import user_data
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere


template = """You are a history assistant having a conversation with a curious child. Sadly you do not have any deep knowledge about things outside of history, when the child ask you any deep question that are not related to history. You kindly reply him by saying "Sorry, I am a history assistant, I do not have a deep knowledge about things that are not related to history. Please ask me history related question, I am happy to help!"

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

chat_cohere = ChatCohere(model="command",temperature=0.6, cohere_api_key="TKhniHiOxRvqfbUKAOfpTV54mw1pnyRlPwdTgqh5")


llm_chain = LLMChain(
    llm=chat_cohere,
    prompt=prompt,
    verbose=True,
    memory=memory,
)




st.title("History Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("What are your curious about...")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content":prompt})

    response = llm_chain.predict(human_input=prompt)
    flow = user_data(function_name=response)

    with st.chat_message("assistant"):
        st.write(flow)
    st.session_state.messages.append({"role": "assistent", "content":response})

