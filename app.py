import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
from nomic import embed

load_dotenv()

# Function to extract text from multiple PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS vector store using Nomic embeddings
def create_vector_store(text_chunks):
    embeddings = embed.text(texts=text_chunks, model="nomic-embed-text-v1.5", inference_mode="local")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")

# Function to load the conversational chain model
def get_conversational_chain():
    prompt_template = """
    You are Taxy, a highly experienced accountant providing tax advice based on Indian Tax laws.
    You will respond to the user's queries by leveraging your accounting and tax expertise and the Context Provided.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3,
                                   system_instruction="You are Taxy, a highly experienced tax advisor.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and fetch the relevant documents for the question
def user_input(user_question):
    embeddings = embed.text(texts=[user_question], model="nomic-embed-text-v1.5", inference_mode="local")
    new_db = FAISS.load_local("Faiss", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main function to handle Streamlit UI and PDF ingestion
def main():
    st.set_page_config("Taxy", page_icon=":scales:")
    st.header("Taxy: AI Tax Advisor :scales:")

    # Load and index PDF documents
    if not os.path.exists("Faiss"):
        pdf_files = []
        for file in os.listdir("dataset"):
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join("dataset", file))

        if pdf_files:
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            create_vector_store(text_chunks)

    # Chat interface
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm Taxy, an AI Tax Advisor."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.write(response)

            if response is not None:
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
