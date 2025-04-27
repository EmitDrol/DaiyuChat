import sys
sys.path.append('/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/.venv/lib/python3.10/site-packages')

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Step 1: 定义索引保存路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = f"{CURRENT_DIR}/index_storage"
DATA_DIR = f"{CURRENT_DIR}/data"
EMBEDDING_MODEL = f"{CURRENT_DIR}/ebmodels/bge-small-zh-v1.5"

def load_and_index_documents(data_dir: str = DATA_DIR):
    """
    加载文档并创建向量索引。
    如果索引已存在，则直接加载；否则创建新索引并保存。
    """
    try:
        # # 检查是否已有持久化索引
        # if os.path.exists(INDEX_DIR):
        #     # 删去持久化索引
        #     os.remove(INDEX_DIR)
        
        # 创建新索引
        print("未检测到索引，正在创建新索引...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        
        # 使用 from_params 方法初始化 ChromaVectorStore
        vector_store = ChromaVectorStore.from_params(
            collection_name="default_collection",
            persist_dir=INDEX_DIR
        )
        
        # 使用嵌入模型创建索引
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        )
        
        # # 持久化索引
        # index.storage_context.persist(persist_dir=INDEX_DIR)
        # print("索引已创建并持久化。")
        
        return index
    except Exception as e:
        raise ValueError(f"加载文档或创建索引失败: {str(e)}")

def retrieve(index, query: str, top_k: int = 5):
    """
    根据用户查询进行语义检索，并返回最相关的文档片段。
    """
    try:
        # 使用索引中的检索器进行语义搜索
        retriever = index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(query)
        
        # 提取检索结果的文本内容
        results = [node.text for node in retrieved_nodes]
        
        return results
    except Exception as e:
        raise ValueError(f"检索失败: {str(e)}")

# 示例使用
if __name__ == "__main__":
    # Step 1: 加载或创建索引（假设文档存放在 'data/' 目录下）
    index = load_and_index_documents()
    
    while True:
        # Step 2: 定义用户查询
        user_query = input("请输入查询：")
        
        # Step 3: 检索相关文档
        results = retrieve(index, user_query, top_k=5)
        
        # Step 4: 打印结果
        print("检索结果：")
        for i, result in enumerate(results, 1):
            print(f"[[{i}]]. {result}")