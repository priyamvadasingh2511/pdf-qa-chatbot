import streamlit as st
from dotenv import load_dotenv
import fitz # PyMuPDF for text extraction
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import uuid
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
#st.write("API Key Loaded:", "HUGGINGFACEHUB_API_TOKEN" in os.environ)


st.title("PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions based on its content!")

uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
#initialize session state variables
#crete a unique session ID for each user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

#load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#Function to extract text from PDF
def extract_text_from_PDF(pdf_path):
    text=""
    doc = fitz.open(pdf_path)
    for page in doc:
        text+= page.get_text("text") + "\n"
    return text

#Function to extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    tables = []#empty list since PDFplumber return a list
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_table = page.extract_table()
             # Convert None cells to empty strings
            if extracted_table:
                tables.append("\n".join(["\t".join(row) for row in extracted_table]))  # Convert table to text
    return tables

#Create a FAISS index and store vectors in it
def create_faiss_index(text_data):
    #divide the text into chunks: will perform better vectorization
    text_chunks = text_data.split("\n\n")
    # show the first chunk
    #st.write("First chunk:", text_chunks[0][:300])
    #embed these chunks of data into vector using embedding model
    #using np.array to convert the vectors into 2D aaray as expected by FAISS
    embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])
     # show embedding shape
    #st.write("Embedding shape:", embeddings.shape)
    #st.write("Sample vector:", embeddings[0][:10])  # first 10 values
    #create FAISS index to store the embeddings
    dim = embeddings.shape[1]  # dimension Faiss needs it to know the size of each vectar it will store
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    #st.write("FAISS total vectors:", index.ntotal)
    return index, text_chunks
    


#Function to Retreive the text 
def retreive_text(query, index, text_chunks, top_k=3):
    query_embeddings = np.array([embedding_model.encode(query)])
    distance, indices = index.search(query_embeddings, top_k) #index.search finds top-k closest vectors to the query vector.
    return [text_chunks[i] for i in indices[0]]#FAISS only gives row numbers, not the original text.This line maps those indices back to the actual text.
                                                #Returns a list of the top relevant chunks.

#Loading the LLM

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)
#st.write("LLM type:", type(llm))
#Prompt to send to LLM
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert AI assistant. Answer strictly based on the provided context. 
    If the context does not contain the answer, reply 'I don't know'.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
)

qa_chain = prompt | llm | StrOutputParser()

#process the uploaded PDF
if uploaded_file :
    st.write("PDF uploaded successfully!")
#creating a unique name for the pdf file with session id
    pdf_path = f"uploaded_file{st.session_state.session_id}.pdf"
    #storing the pdf in disk
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    #Extract text and tables
    extracted_text = extract_text_from_PDF(pdf_path)
    extracted_tables = extract_tables_from_pdf(pdf_path)
    #combine text and tables
    all_text_data = extracted_text + "\n".join(extracted_tables)

    #create FAISS index and store in session state
    index, text_chunks = create_faiss_index(all_text_data)
    #storing them in session state
    st.session_state.faiss_index = index
    st.session_state.text_chunks = text_chunks
    st.success("PDF processed! You can now ask questions.")

#User question 
user_question = st.text_input("Ask a question about the PDF")

if user_question and st.session_state.faiss_index:
    #Retrieve relevant chunks
    relevant_chunks= retreive_text(user_question, st.session_state.faiss_index, st.session_state.text_chunks)
    #relevant_chunks will retrun string of text chunks which we need to cobmine in a single line befire sending to LLM
    context = "\n".join(relevant_chunks)  
    #st.write("Relevant chunks:", relevant_chunks)
    #st.write("Context preview:", context[:500]) 
    response = qa_chain.invoke({"context": context, "question": user_question})
    answer = response  
    #st.write("Relevant chunk indices:", relevant_chunks)

    st.write("**Answer:**")
    st.write(answer)
elif uploaded_file:
    st.info("Ask a question about the PDF.")

else:   
    st.warning("Please upload the PDF")

   




