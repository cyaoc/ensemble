import shutil
import os
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.llms import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def engine_building(input_file,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_api_version):
    llm = AzureOpenAI(
        engine=engine,
        model="gpt-35-turbo-16k",
        temperature=0.3,
        max_tokens=4096,
        api_base=api_base,
        api_key=api_key,
        api_type="azure",
        api_version=api_version,
    )
    embed_model = OpenAIEmbedding(
        model_name=embed_model_name,
        deployment_name=embed_deployment_name,
        api_key=api_key,
        api_base=api_base,
        api_type="azure",
        api_version=embed_api_version,
    )
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes=nodes,service_context=service_context)
    folder_name = "storage"
    shutil.make_archive(folder_name, "zip", folder_name)
    return index.as_query_engine(similarity_top_k=5), f"{folder_name}.zip"

def engine_loading(input_file,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_api_version):
    file_dir = os.path.dirname(input_file)
    folder = os.path.join(file_dir, "storage")
    shutil.unpack_archive(input_file, folder, "zip")
    llm = AzureOpenAI(
        engine=engine,
        model="gpt-35-turbo-16k",
        temperature=0.3,
        max_tokens=4096,
        api_base=api_base,
        api_key=api_key,
        api_type="azure",
        api_version=api_version,
    )
    embed_model = OpenAIEmbedding(
        model_name=embed_model_name,
        deployment_name=embed_deployment_name,
        api_key=api_key,
        api_base=api_base,
        api_type="azure",
        api_version=embed_api_version,
    )
    service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    storage_context = StorageContext.from_defaults(persist_dir=folder)
    index = load_index_from_storage(storage_context, service_context=service_context)
    return index.as_query_engine(similarity_top_k=10)