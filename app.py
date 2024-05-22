import streamlit as st
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from unstructured.cleaners.core import group_broken_paragraphs, clean_non_ascii_chars, clean_extra_whitespace, clean_bullets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

if not 'valid_inputs_received' in st.session_state:
  st.session_state['valid_inputs_received'] = False

if not 'process' in st.session_state:
  st.session_state['process'] = False

def load_files(pdf_docs):
  pages = []
  for pdf in pdf_docs:
    tmp_file = f"./{pdf.name}"
    with open(tmp_file, 'wb') as file:
    #  print(pdf.getvalue())
      file.write(pdf.getvalue())
      loader = PyPDFLoader(tmp_file)
      pages = [*pages, *loader.load_and_split()]
  #print(os.curdir)
    os.remove(tmp_file)
  return pages

def chunk_document(raw_text):
  def clean_page_content(documents):
    for document in documents:
      document.page_content = clean_non_ascii_chars(document.page_content)
      document.page_content = clean_extra_whitespace(document.page_content)
      document.page_content = group_broken_paragraphs (document.page_content)
      #document.page_content = clean_dashes(document.page_content)
      document.page_content = clean_bullets(document.page_content)
      #document.page_content = clean_extra_whitespace(document.page_content)
      #document.page_content = TextBlob(document.page_content).correct().raw

  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
  clean_page_content(raw_text)
  split_documents = splitter.split_documents(raw_text)
  return split_documents

@st.cache_resource(show_spinner="Loading VectorStore...")
def load_vector_store(api_key):
  """Load a vector store from a Weights & Biases artifact
  Args:
      run (wandb.run): An active Weights & Biases run
  Returns:
      Chroma: A chroma vector store object
  """
  #embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
  # load vector store
  vector_store = Chroma(
      embedding_function=OpenAIEmbeddings()
  )

  return vector_store

def create_prompt():
  template = {"system":"""
    You are a career adviser helping people make an introduction based on their CV to help them get a job.
    When asked to compose an introduction, use the provided context write it in first person.
    If they ask questions, only use the context as reference, do not make things up.
    {context}
  """,
  "human":"""
    <<<
    Chat history: {chat_history}
    Request: {question}
    >>>
  """}
  messages = [
        SystemMessagePromptTemplate.from_template(template["system"]),
        HumanMessagePromptTemplate.from_template(template["human"]),
    ]
  prompt = ChatPromptTemplate.from_messages(messages)
  return prompt

def load_chat_space(chain):
  for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])


  response = {}
  if prompt := st.chat_input('Start chatting'):
    with st.chat_message('human'):
      st.markdown(prompt)
    with st.chat_message('assistant'):
      with st.spinner('Thinking...'):
        response = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history},
          return_only_outputs=True,
        )
      st.write(response['answer'])#TODO: Fix rendering of reply
      #st.write(response['source_documents'])#TODO: Fix rendering of reply
    #print(response[0]['answer'])
    st.session_state.chat_history.append((prompt, response['answer']))
    st.session_state.conversation.append({'role':'human', 'content':prompt})
    st.session_state.conversation.append({'role':'assistant', 'content':response['answer']})

def main():
  def is_api_key_valid(submitted, api_key):
    if not submitted and not st.session_state.valid_inputs_received:
      return False
    elif submitted and not api_key:
      st.warning(f"Please input your API key")
      st.session_state.valid_inputs_received = False
      return False
    else: # submitted and api_key
      if not api_key.startswith('sk-'):
        st.warning(f'Please enter your OpenAI API key!', icon='⚠')
        st.session_state.valid_inputs_received = False
        return False
      else:
        st.session_state.valid_inputs_received = True
        return True
  
  st.title("Introduction Generator using CV 📃")
  if "conversation" not in st.session_state:
    st.session_state.conversation = [{'role':'assistant', 'content':'How may I help you?'}]
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
  
  split_text = ""
  process = None
  with st.sidebar:
    with st.form('api_form'):
      api_key = st.text_input(label="API Key", placeholder=f"Enter your API Key", type="password")
      submit_button = st.form_submit_button(label="Submit")
    if not is_api_key_valid(submit_button, api_key):
      st.stop()

    st.subheader('Your work documents:')
    docs = st.file_uploader(
      'Please upload your PDF CV and click process.',
      type='pdf',
      accept_multiple_files=True)
    if len(docs) == 0:
      st.button('Process', disabled=True, use_container_width=True)
    else:
      process = st.button('Process', use_container_width=True)

  if process:
    with st.spinner("Processing"):
      #Parse document to load to vector store
      #print(pdf_docs[0])
      raw_text = load_files(docs)
      split_text = chunk_document(raw_text)
      #Retrieve vector store
      vector_store = load_vector_store(api_key)
      with st.spinner("Combining documents..."):
        vector_store.add_documents(documents=split_text)
      #Load Model
      user_prompt = create_prompt()
      llm=ChatOpenAI(openai_api_key=api_key, temperature=0.3, callbacks=[StreamingStdOutCallbackHandler()])
      chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs=dict(k=5)),
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": user_prompt},
        return_source_documents=False,
      )
      st.session_state.process = process
      st.session_state.chain = chain
  if st.session_state.process and st.session_state.chain:
    load_chat_space(st.session_state.chain)

if __name__ == '__main__':
  st.set_page_config(page_title="Introduction Generator using CV",
                      page_icon="📃")
  main()