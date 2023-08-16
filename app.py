from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import json
from render import user_msg_container_html_template, bot_msg_container_html_template
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import glob
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from pandasai import PandasAI
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import OpenAI


def submit():
    st.session_state.input = st.session_state.widget
    st.session_state.widget = ''

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

    
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF",initial_sidebar_state="expanded")
    st.header("Ask your PDFs and CSVs💬")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Add custom CSS styles to change the background color
    st.markdown(
        """
        <style>
        .main {
            background-color: #98D7C2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # upload file
    uploaded_files = st.file_uploader(
            "Upload your PDFs or CSVs here", accept_multiple_files=True)
            
    # Create a temporary directory to store the uploaded files
    temp_dir = tempfile.mkdtemp()
    temp_path = temp_dir

    # Save the uploaded files to the temporary directory
    for i, uploaded_file in enumerate(uploaded_files):
        # Get the filename from the uploaded file
        filename = uploaded_file.name

        # Extract the file extension using os.path.splitext
        file_extension = os.path.splitext(filename)[1]
        file_path = os.path.join(temp_path, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

    # Get all CSV files in the temporary directory
    csv_files = glob.glob(os.path.join(temp_path, "*.csv"))
    pdf_docs = glob.glob(os.path.join(temp_path, "*.pdf"))

    # Create an empty dictionary to store the DataFrames
    dataframes = {}

    # Iterate over the CSV files
    for file in csv_files:
        # Extract the file name without extension
        file_name = file.split('/')[-1].split('.')[0]
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Store the DataFrame in the dictionary
        dataframes[file_name] = df
            
    raw_text = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    
    if text_chunks:
    # create vector store
        knowledge_base = get_vectorstore(text_chunks)

            
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'input' not in st.session_state:
        st.session_state.input = ''
            
    user_question = st.text_input("Ask your question and click on the LLM model you want to use:",key='widget', on_change=submit)

    button_clicked_1 = st.button("gpt-4")
    button_clicked_2 = st.button("gpt-3.5-turbo")
    #button_clicked_3 = st.button("text-davinci-003")
    
    # extract the text
    if button_clicked_1:
        try:
            docs = knowledge_base.similarity_search(st.session_state.input)
        
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.1, model="gpt-4"),
                knowledge_base.as_retriever()
            )
            with get_openai_callback() as cb:
              response = qa({"question": st.session_state.input, "chat_history": st.session_state.chat_history})
              st.session_state.chat_history.append((st.session_state.input, response["answer"]))
              
        except:
        
            llm =  pandasai.llm.OpenAI()
            # create PandasAI object, passing the LLM
            #pandas_ai = PandasAI(llm, conversational=False, verbose=True)
            #pandas_ai.clear_cache()
            df = SmartDatalake(list(dataframes.values()), config={"llm": llm})
        
            if any(word in st.session_state.input for word in ["plot","chart","Plot","Chart"]):
                question = st.session_state.input + ' ' + 'using seaborn'
            else:
                question = st.session_state.input

            #fig = go.Figure()
            fig = plt.gcf()
            #x = pandas_ai.run(list(dataframes.values()), question)
            x = df.chat(question)
            if fig.get_axes() > 0:
                st.session_state.chat_history.append((st.session_state.input, fig))
            
            else:
                st.session_state.chat_history.append((st.session_state.input, x))
            
            
          
    elif button_clicked_2:
        try:
            docs = knowledge_base.similarity_search(st.session_state.input)
        
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo"),
                knowledge_base.as_retriever()
            )
            with get_openai_callback() as cb:
              response = qa({"question": st.session_state.input, "chat_history": st.session_state.chat_history})
              st.session_state.chat_history.append((st.session_state.input, response["answer"]))
              
        except:
            llm = OpenAI()
            # create PandasAI object, passing the LLM
            pandas_ai = PandasAI(llm, conversational=False, verbose=True)
            #pandas_ai.clear_cache()
        
            if any(word in st.session_state.input for word in ["plot","chart","Plot","Chart"]):
                question = st.session_state.input + ' ' + 'using seaborn'
            else:
                question = st.session_state.input

            #fig = go.Figure()
            fig = plt.gcf()
            x = pandas_ai.run(list(dataframes.values()), question)

            if fig.get_axes():
                st.session_state.chat_history.append((st.session_state.input, fig))
            
            else:
                st.session_state.chat_history.append((st.session_state.input, x))
                
    # Display chat history
    for message in st.session_state.chat_history[::-1]:
        if message[0]:
            st.write(user_msg_container_html_template.replace("$MSG", message[0]), unsafe_allow_html=True)
            try:
                st.pyplot(message[1])
            except:
                st.write(bot_msg_container_html_template.replace("$MSG", str(message[1])), unsafe_allow_html=True)
            
if __name__ == '__main__':
    main()

