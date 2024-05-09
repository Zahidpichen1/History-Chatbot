import streamlit as st
from Functions.write_stream import user_data
from langchain_community.llms import LlamaCpp
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, tool, DuckDuckGoSearchRun

##############==============  BACKEND  ==============##############

llm = LlamaCpp(
    model_path="Model\mistral-7b-v0.1.Q5_K_S.gguf",
    temperature=0,
    max_tokens=3000,
    n_ctx = 1100,
    verbose = True
    )


conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

search = DuckDuckGoSearchRun()

stool = Tool.from_function(
    func =search.run,
    name = "Browzing",
    description = "A useful search engine used only for topics related to history. Using this serach engine, it is possible to get latest news related to history topics"
)

tools = [stool]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    max_iterations=5,
    verbose = True,
    memory=conversational_memory
)

##############==============  UI(STREAMLIT APP)  ==============##############
st.title("History Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
prompt = st.chat_input("Enter your message: ")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({'role':"user",'content':prompt})

    output = agent(prompt)
    response =  output['output']
    flow = user_data(function_name= response)
    with st.chat_message("assistant"):
        st.write_stream(flow) 

    st.session_state.messages.append({'role':"assistant",'content': response})


with st.sidebar:
    New_Chat = st.button("New Chat", use_container_width=True)
    if New_Chat:
        # Delete all the items in Session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.messages = []  # Re-initialize the messages list
