
import time
import psutil
import glob
import PyPDF2
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline
)
from transformers import LlamaTokenizer, LlamaForCausalLM,BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from llama_cpp import Llama



def RAG_Chain(pdf_file,question):
    model_path = "/home/mona/Downloads/Pubmed_model_GGUF"
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    doc = ""
    for page_num in range(len(pdf_reader.pages) ):
        page = pdf_reader.pages[page_num] 
        doc += page.extract_text()
    # Check if any documents were loaded
    if not doc:
        raise ValueError("No documents found. Please check the PDF directory path.")
    # Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(doc)

    # Create HuggingFace embeddings and vector store
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Efficient model suitable for most tasks
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)

    '''
        # Load the LLM model
    llm_model_name =  'PubMed_Llama3.1_Based_model'  # Replace with any suitable HuggingFace model

        # tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

        ## Load the tokenizer and model
    
    try:
            # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Attempting to use LlamaTokenizer and LlamaForCausalLM instead.")
    
    
    try:
        tokenizer = LlamaTokenizer.from_pretrained(llm_model_name)
        model = LlamaForCausalLM.from_pretrained(llm_model_name)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the model and tokenizer are compatible and the 'transformers' library is up to date.")
    '''  
    
    #base_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    '''
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        trust_remote_code=True,  # Required for some models
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,  # Specify 4-bit quantization within BitsAndBytesConfig
            load_in_8bit_fp32_cpu_offload=True  # Enable CPU offload
        )
    )
    
    #adapter_path = "mohamedalcafory/PubMed_Llama3.1_Based_model"
  
    model.load_adapter(model_path)          
    # Create the HuggingFace pipeline
    hf_pipeline = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=False
        )
'''
    # Wrap the pipeline in a LangChain LLM
    #llm = HuggingFacePipeline(pipeline=hf_pipeline)
  

    # Instantiate the Llama model using the gguf file

    llama_model = Llama(
        model_path,
        n_ctx=2048,          # Context length
        n_threads=8,         # Number of CPU threads to use
        temperature=0.7      # Sampling temperature
    )



    # Define the retriever using Chroma
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return "No relevant information found in the documents."
    # Format the context
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # Prepare the prompt for the LLM
    formatted_prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{formatted_context}\n\nQuestion: {question}\n\nAnswer:"
    )
    # Generate the answer
    answer = llama_model(formatted_prompt)
    return answer

def monitor_resources():
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    return cpu_usage, memory_usage

def get_important_facts(question):
    start_time = time.time()
    cpu_before, mem_before = monitor_resources()
    answer = rag_chain(question)
    cpu_after, mem_after = monitor_resources()
    cpu_diff = cpu_after - cpu_before
    mem_diff = mem_after - mem_before
    end_time = time.time()
    response_time = end_time - start_time
    print(
        f"Response Time: {response_time:.2f} seconds, "
        f"Memory usage: {mem_diff:.2f} MB, CPU usage: {cpu_diff:.2f}%"
    )
    return answer
'''
# Create a Gradio app interface
iface = gr.Interface(
    fn=get_important_facts,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Chatbot",
    description="Ask questions about the content in your PDFs",
)

# Launch the Gradio app
iface.launch(debug=True)
'''