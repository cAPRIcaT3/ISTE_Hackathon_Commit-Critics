from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from torch import cuda

def setup_rag_pipeline():
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    loader = WebBaseLoader("https://www.example-knowledge-base.com/")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)

    llm = LlamaCpp(
        model_path="/path/to/llama-2-13b-chat.ggmlv3.q5_1.bin",
        n_gpu_layers=32,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
    )

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='default',
        retriever=vectorstore.as_retriever()
    )

    return rag_pipeline

def generate_comment(query):
    rag_pipeline = setup_rag_pipeline()
    response = rag_pipeline(query)
    return response
