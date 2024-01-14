# PDFGemini_Bot

## Project Overview

PDFGem Chat is an interactive chat interface designed for querying information from uploaded PDF files. This project utilizes Streamlit, PyPDF2, LangChain, Google Generative AI, and FAISS to create a seamless experience for users to ask questions related to the content of PDF documents.

<img width="960" alt="image" src="https://github.com/SurajSanap/PDFGemini_Bot/assets/101057653/15fe59a5-cf8a-4b9b-8e84-536a99bf1cfe">

## Components

### 1. User Interface

- Developed using the Streamlit library for a user-friendly experience.
- Users can ask questions about the content of uploaded PDF files.

### 2. PDF Processing

- Extracts text from PDF files using PyPDF2.
- Splits extracted text into manageable chunks.

### 3. Embedding and Vectorization

- Leverages Google Generative AI Embeddings for converting text into vectors.
- Applies FAISS (Facebook AI Similarity Search) to create a vector store/index of text chunks.

### 4. Conversational Chain

- Implements a conversational chain for question-answering using the Gemini Generative AI model.
- Configures the prompt template for providing context and framing questions.

### 5. Workflow

- Users upload PDF files and ask questions through the interface.
- Text is extracted from PDFs, split into chunks, and converted into vectors.
- The conversational chain processes user input, searches for similar text chunks, and generates responses.

## Code Structure

- **`main()` function**: Sets up the Streamlit interface and handles user input.
- **`get_pdf_text(pdf_docs)` function**: Extracts text from PDF files.
- **`get_text_chunks(text)` function**: Splits text into manageable chunks.
- **`get_vector_store(text_chunks)` function**: Creates a vector store/index from text chunks.
- **`get_conversational_chain()` function**: Configures the conversational chain for question-answering.
- **`user_input(user_question)` function**: Processes user input and generates responses.
- **Environment variables**: Utilizes the `dotenv` library to securely load the Google API key.

## Usage

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files.
2. **Ask a Question**: Enter your question in the provided text input.
3. **Submit & Process**: Click the button to initiate the processing of PDFs and question-answering.
4. **View Response**: The system generates a response based on the input question and the content of the PDFs.

## Dependencies

- Streamlit
- PyPDF2
- LangChain
- Google Generative AI
- FAISS
- Dotenv

## Setup

1. **Install Dependencies**: Ensure the required Python packages are installed.
2. **Set up Google API Key**: Store the Google API key in a secure manner using the `dotenv` file.
3. **Run the Application**: Execute the script to launch the Streamlit interface.

Feel free to explore and enhance the functionalities of this project based on your requirements. 

---

**PDFGemini** - Unleash the Power of Conversational PDF Exploration! 💬✨
