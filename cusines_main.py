import os
import time
import pickle
import streamlit as st
import langchain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
import re
import requests
from langchain_community.document_loaders.csv_loader import CSVLoader
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import textwrap
from IPython.display import Image
from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
global_counter = 0
def save_image_from_url(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved successfully to {save_path}")
        else:
            print("Failed to download image. Check the URL and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def ImageGen(entity, count):
    global global_counter
    global_counter += 1
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate a detailed prompt of 100 words only to generate an image based on the following description: {image_desc}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    image_url = DallEAPIWrapper().run(chain.run("Generate a pencil sketch of each of its ingredients of " + entity + " without their names in a single image. Do not add any text in the image"))
    save_path = "{}_{}.jpg".format(count, global_counter)
    save_image_from_url(image_url, save_path)

def preprocess_data(data):
    modified_str = ""
    for i in range(len(data)):
        unmodified_str = data[i].page_content
        partially_modified_str = re.sub(r'\n+', '\n', unmodified_str)
        partially_modified_str = re.sub(r'\t+', '\t', partially_modified_str)
        modified_str = re.sub(r' +', ' ', partially_modified_str)
        data[i].page_content = modified_str

with open('.\LLM Project\openAiKey.txt', 'r') as file:
    os.environ['OpenAI_API_KEY'] = file.read().strip()

st.title("Cuisines Info Dataset")
urls = []
# for i in range(2):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

main_placefolder = st.empty()
query_placefolder = st.empty()
process_url_clicked = st.button("Process URLs")

if process_url_clicked:
    # loader = SeleniumURLLoader(urls=urls)
    loader = CSVLoader(file_path="./LLM Project/Cuisines-NLP/cuisines.csv", source_column="Link")
    main_placefolder.text("Data loading started... ")
    data = loader.load()
    # data = preprocess_data(data)
    # Split documents
    text_splitter_module = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"],
        chunk_size=2000,
        chunk_overlap=200
    )
    documents = text_splitter_module.split_documents(data)
    main_placefolder.text("Data splitted into docs... ")
    embeddings = OpenAIEmbeddings()
    # faiss_indices = FAISS.from_documents(documents, embeddings)
    # faiss_indices.save_local("faiss_indices_for_app")

query = query_placefolder.text_input("Enter your question here: ")

if query:
    if os.path.exists("./faiss_indices_for_app/index.pkl"):
        faiss_index = FAISS.load_local("faiss_indices_for_app", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        main_placefolder.text("Embeddings generated... ")
        llm = OpenAI(temperature=0.9, max_tokens=500)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_index.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            count = 2
            for source in sources_list:
                if count > 0:
                   ImageGen(source, count)
                   image_path = "{}_{}.jpg".format(count, global_counter)
                   st.image(image_path, caption=source)
                # st.write(source)
                count -= 1


        main_placefolder.text("Generated sucessfully... ")
        # if st.button("Refresh and Try again"):
        #     streamlit_js_eval(js_expressions="parent.window.location.reload()")
