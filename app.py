import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

st.set_page_config(page_title="chat with pdf",layout="centered")
st.title('DocMate')
st.caption('Converse with any PDF with the power of RAG!')

llm = Ollama(model="mistral:latest")

embeddings = OllamaEmbeddings(model="mistral")

parser = StrOutputParser()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])    

# Function to get the file path from the user
def get_file_path():
    path=st.text_input("Please enter the path to your PDF file",key=2)
    return path

@st.cache_resource(show_spinner=True)
def load_vectorDB(file_path):
    #splitting the pages
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    st.caption("file loaded sucessfully")
    
    vector_database = Chroma.from_documents(pages, embedding=embeddings)
    
    retriever = vector_database.as_retriever()
    return retriever
            
        
prompt_template = """
You are an intelligent AI assistant named DocMate. Your role is to thoroughly read the given context from a PDF document and
then provide clear, accurate and helpful answers to any questions based on that context. If you cannot find enough relevant information in the 
context to answer a question, reply with "I'm sorry, I don't have enough information to answer that question."

Context: {context}

Question: {question}      

"""
prompt=PromptTemplate.from_template(prompt_template)

try:

    file =get_file_path()
    if file:        

        retriever1 = load_vectorDB(file)
        st.caption("vector database created successfully")

        chain=(
        {
            "context": itemgetter("question")|retriever1,
            "question":itemgetter("question")
        }
        | prompt
        | llm
        | parser
        )  

        prompt = st.chat_input("Ask me anything...")
        with st.spinner("Generating response.."):
            # Accept user input
            if prompt :
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                #providing user query to the llm
                response = chain.invoke({"question": prompt})
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        

except Exception as e:
    st.warning(e)
    