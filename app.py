from dotenv import load_dotenv
load_dotenv()  

import streamlit as st
import os
import google.generativeai as genai
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Configure the Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-001")

def get_gemini_response(question, context=None):
    """Generates a response from the Gemini model with optional context."""
    if context:
        question = f"{question}\n\nContext:\n{context}"
    response = model.generate_content(question)
    return response.text

def save_history(question, response):
    """Saves question and response in session state."""
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"question": question, "response": response})

def read_pdf(file):
    """Reads and extracts text from a PDF file."""
    try:
        pdf_reader = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_reader.page_count):
            page = pdf_reader.load_page(page_num)
            text += page.get_text("text")
        pdf_reader.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# New Function to handle PDF text extraction and chunking
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        chunks = []
    return chunks

# Function to create vector store using Google Generative AI embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Function to load the QA conversational chain
def get_conversational_chain():
    prompt_template = """
    
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    say "Answer is not available in the context." Do not provide a wrong answer.

    Context:\n {context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_query(user_question, pdf_text=None):
    """Handles user query with optional PDF context."""
    response = None
    try:
        # If there is a PDF context, process and save embeddings
        if pdf_text:
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.write("No relevant information found in the documents.")
            return "No relevant information found in the documents."

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Return the raw text of the response
        return response["output_text"]
    
    except Exception as e:
        st.error(f"Error processing user query: {e}")
        return "Error processing your query."

# Streamlit app configuration
st.set_page_config(page_title="Q&A Demo with PDF Reader")
st.header("DREJ: The Chat Bot")

# Apply custom CSS for the sidebar
st.markdown(
    """
    <style>
    /* Sidebar background styling */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#122D5C,#122D5C);  /* Gradient background */
        color: white;  /* White text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "first_question_asked" not in st.session_state:
    st.session_state.first_question_asked = False
if "first_question" not in st.session_state:
    st.session_state.first_question = None
if "first_response" not in st.session_state:
    st.session_state.first_response = None

# PDF uploader section
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
pdf_text = None
if uploaded_pdf:
    pdf_text = read_pdf(uploaded_pdf)

# Form to handle input submission with Enter key
with st.form(key='question_form'):
    input_text = st.text_input("Input your question: ", key="input", placeholder="Type your question here...")
    submit_button = st.form_submit_button("Ask the Question")

    if submit_button:
        if input_text.strip():  # Check if input is not empty or just spaces
            # If PDF is uploaded, process the text, otherwise ask the question directly
            if pdf_text:
                response = handle_user_query(input_text, pdf_text)
            else:
                response = get_gemini_response(input_text)
            
            if response:
                st.subheader("The response is")
                st.write(response)
            
            # Add the first question to history only if it is not the first question
            if st.session_state.first_question_asked:
                save_history(st.session_state.first_question, st.session_state.first_response)

            # Save the current question and response
            st.session_state.first_question_asked = True
            st.session_state.first_question = input_text
            st.session_state.first_response = response

        else:
            st.warning("The input cannot be empty. Please enter a question.")

# Sidebar with an image
with st.sidebar:
    try:
        st.image("DREJLOGO.png", use_column_width=False, width=150)  # Adjust width as needed
    except Exception as e:
        st.error(f"Error displaying image: {e}")

    st.subheader("History of Questions")
    if "history" in st.session_state and st.session_state.history:
        history = st.session_state.history
        for i, item in enumerate(reversed(history), 1):
            st.write(f"**Question {len(history) - i + 1}:** {item['question']}")
            st.write(f"**Response {len(history) - i + 1}:** {item['response']}")
    else:
        st.write("No history available.")
