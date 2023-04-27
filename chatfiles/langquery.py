import os
import time

from pytest import console_main
key = os.environ["OPENAI_API_KEY"]
searchkey = os.environ["SERPAPI_API_KEY"]

from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

def test():
    # 加载文件夹中的所有txt类型的文件
    # loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
    # 根据文件类型来定义一个loader，不同的loader能够解析不同的文件内容，最终都会解析为一个大文本
    file_path = 'E:/tempdown/fz.pdf'
    #file_path = 'E:/tempdown/xuexi.pdf'
    
    parent_dir = os.path.dirname(file_path)
    #persist_dir = os.path.join(parent_dir, "vector_store")    
    persist_dir = os.path.join(parent_dir, os.path.basename(file_path) + "_vector_store_dir")
    
    # 加载数据
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(persist_dir):
        docsearch = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        start_time = time.time()        
        print("开始分析文档，时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        loader = file_path.endswith(".pdf") and PyPDFLoader(file_path) or TextLoader(file_path)
        # 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
        #splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "，", "\n"])
        chunks = loader.load_and_split(splitter)

        # 将数据转成 document 对象，每个文件会作为一个 document
        # documents = loader.load()

        # 初始化加载器
        #text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, persist_directory="./vector_store/")
        #text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # 切割加载的 document
        #split_docs = text_splitter.split_documents(documents)
        split_docs = splitter.split_documents(chunks)

        # 初始化 openai 的 embeddings 对象
        #embeddings = OpenAIEmbeddings()
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        #console_main.log('OpenAIEmbeddings')
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
        docsearch.persist()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print("执行时长：%02d:%02d:%02d" % (hours, minutes, seconds))
    # 创建问答对象
    #qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)#vectorstore=docsearch,return_source_documents=True)
    chain2 = get_chain(docsearch)   

    # 本地搜索到的chunk会作为context，和问题一起提交给LLM来处理。我们当然要使用ChatGPT模型了，比GPT-3.0又好又便宜
    llm = ChatOpenAI(temperature=0)    

    # chain是LangChain里的概念，其实就相当于定义了一个流程，这里我们提供的参数就是文档语义搜索工具以及LLM
    chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever())
    #chain1 = load_qa_chain(llm, chain_type="stuff", verbose=True)

    # 进行问答
    # 下面就比较简单了，不断读取问题然后执行chain
    while True:
        question  = input("\nQ: ")
        if not question:
            break
        #print("A:", chain.run(question))    
        #ret = qa({"query": question})
        #ret = chain.run(question)
        #print("A:", ret)
        ret = chain2({"question": question})
        print("A:", ret['answer'])
        '''
        result = ""
        for document in ret['source_documents']:
            page = document.metadata['page']
            result += f", {page}"
        result = result.lstrip(", ")
        result = ret['result'] + (f"\n内容所在页码：{result}" if result else "")
        print("A:", result)
        '''
        #chain1.run(input_documents=docsearch, question=question)
        

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
system_template="""Answer the user's question in Chinese using the following context snippets. 
If there is no relevant information mentioned in the document based on the following context, just say '文档中没有提到相关内容'.
If you don't know the answer, just say "嗯..., 我不知道答案.", don't try to make up an answer.
ALWAYS return a "Sources" part in your answer.
The "Sources" part should be a reference to the source of the document from which you got your answer.
Example of your response should be:
```
The answer is foo
Sources:
1. abc
2. xyz
```
Begin!
----------------
{summaries}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

def get_chain(store):
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0.2), 
        chain_type="stuff", 
        retriever=store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )
    return chain

test()