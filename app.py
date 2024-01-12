import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from transformers import pipeline

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context and from internet also, make sure to provide all the details, if the answer is not in
    provided context just say, search on google. You can include Markdown font, bullets an tables if required.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question, context_docs):
    question_answering = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    answers_container = st.empty()

    for doc in context_docs:
        result = question_answering(question=user_question, context=doc)
        answers_container.write("Answer:", result["answer"])




def main():
    st.set_page_config("SRS-Gemini")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Start Train") and pdf_docs:
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text:
                        st.warning("No text found in the uploaded PDFs. Please upload valid PDF files.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)

                    context_docs = get_text_chunks(raw_text)  # Each page becomes a context document
                    user_input(user_question, context_docs)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    # Log the error for further investigation
                    st.exception(e)

            st.success("Done")
