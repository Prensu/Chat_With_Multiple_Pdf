import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS 
from langchain.embeddings import OpenAIEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",           
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings= HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings) 
    return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon="📚")

    st.header("📚 Chat With Multiple PDFs")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    
                    st.success("✅ Text successfully processed and embedded!")
                    st.write("📄 Text Preview:")
                    st.text(raw_text[:1000])
            else:
                st.warning("⚠️ Please upload at least one PDF.")

if __name__ == "__main__":
    main()
