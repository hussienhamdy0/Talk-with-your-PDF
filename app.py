import streamlit as st
import PyPDF2
from io import StringIO
#import PyPDF2
from RAG_GGUF import RAG_Chain
from llama_cpp import Llama



# Function to send pdf file to RAG pipeline
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages) ):
        page = pdf_reader.pages[page_num] 
        text += page.extract_text()
    return text

st.title("Talk with Your PDF")

# PDF Upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
        # Display the file name
    st.write(f"File uploaded: {uploaded_file.name}")
    # Read and display the content of the uploaded PDF file
    try:
        pdf_content = read_pdf(uploaded_file)
        st.text_area("PDF Content", pdf_content, height=300)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
# Input field for user messages
    user_input = st.text_input("You:", "")
else:
 st.text_area("PDF Content","Please Upload File",height=300)



# Initialize a session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
model_path = "/home/mona/Downloads/Pubmed_model_GGUF"
llama_model = Llama(
        model_path,
        n_ctx=2048,          # Context length
        #n_threads=8,         # Number of CPU threads to use
        temperature=0.7,      # Sampling temperature
        n_gpu_layers=4
)

# Handle user input
if st.button("Send"):
    #import chromadb.api
    #chromadb.api.client.SharedSystemClient.clear_system_cache()
    if user_input:
        # Get the GPT response
        gpt_response = RAG_Chain(uploaded_file,user_input,llama_model)

        # Store the conversation
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("BOT", gpt_response))

        # Clear the input box
        user_input = ""

# Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:** {message}")
  

    








