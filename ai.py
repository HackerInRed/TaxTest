import streamlit as st
from nomic import embed
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

def get_conversational_chain():
    prompt_template = """
    You are Taxy, a highly experienced accountant providing tax advice based on Indian Tax laws.
    You will respond to the user's queries by leveraging your accounting and tax expertise and the Context Provided.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, system_instruction="You are Lawy, a highly experienced attorney providing legal advice based on Indian laws.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = embed.text(texts=[user_question], model="nomic-embed-text-v1.5", inference_mode="local")
    new_db = FAISS.load_local("Faiss", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("Taxy", page_icon=":scales:")
    st.header("Taxy: AI Tax Advisor :scales:")
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