📘 PDF Q&A Chatbot

A Streamlit-based application that allows users to upload a PDF and ask questions related to its content. The system extracts text and tables from the PDF, creates semantic embeddings, performs vector search using FAISS, and generates answers using an LLM hosted on Hugging Face Inference API.

🚀 Features

📄 PDF Upload Support
Users can upload any PDF and extract text + tables.

🔍 Semantic Search with FAISS
Relevant chunks are retrieved using vector similarity.

🤖 LLM-Powered Answers
Uses the Llama 3.2 3B Instruct model (free on Hugging Face) via HuggingFaceEndpoint.

🧠 Embeddings with Sentence Transformers
Uses all-MiniLM-L6-v2 for lightweight, accurate vector embeddings.

🧾 Transparent Debug Info (Optional)
Shows retrieved text chunks and context preview for debugging.

🧱 Streamlit UI
Simple, responsive interface for interactive question answering.

📂 Project Structure
📁 project-root/
│── app.py                # Main Streamlit application
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
│── .env                  # Environment variables (API keys)

🛠️ Technologies Used
Component	Technology
UI	Streamlit
PDF Parsing	PyMuPDF (fitz), pdfplumber
Embeddings	sentence-transformers
Vector DB	FAISS
LLM	HuggingFaceEndpoint (Llama 3.2 3B Instruct)
Environment	Python 3.11, virtualenv/pyenv
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/priyamvadasingh2511/pdf-qa-chatbot.git
cd pdf-qa-chatbot

2️⃣ Create & activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate     # Mac / Linux
venv\Scripts\activate        # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your Hugging Face API Token

Create a .env file:

HUGGINGFACEHUB_API_TOKEN=your_token_here


Get your token here:
https://huggingface.co/settings/tokens

5️⃣ Run the application
streamlit run app.py


Open browser at:
http://localhost:8501

🧠 How It Works (Architecture)

User uploads PDF
→ App extracts raw text using PyMuPDF + tables using pdfplumber.

Text is chunked and embedded
→ Embeddings generated using all-MiniLM-L6-v2.

FAISS Index is built
→ Enables fast similarity search.

User asks a question
→ Embedding of question is generated, FAISS returns top-K chunks.

LLM receives prompt containing

Retrieved context

User question

LLM generates answer using HuggingFace model.

📝 Example Prompt Sent to LLM
You are an expert AI assistant. Answer strictly based on the provided context. 
If the context does not contain the answer, reply 'I don't know'.

Context:
<retrieved PDF text>

Question:
<user question>

Answer:

📊 Debug Information

The app shows:

Relevant text chunks

Context preview sent to the LLM

Embedding shapes

Sample embedding vector

This helps validate whether:

The PDF was parsed correctly

FAISS indexing works

Retrieval is accurate

🔒 Environment Variables
Variable	Description
HUGGINGFACEHUB_API_TOKEN	Token to access Hugging Face Inference API
🚧 Future Enhancements

🔍 Highlight PDF text used to generate answer

🗂️ Use better chunking (overlapping windows)

📄 Support multiple PDFs at once

🤖 Option to switch between LLMs

🧩 Add streaming responses

🤝 Contributing

Contributions are welcome!
Please open an issue or submit a pull request.
