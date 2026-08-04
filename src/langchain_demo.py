import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader_text = open("data/sample.pdf", "rb")  # placeholder, real loader below

from loader import load_pdf
text = load_pdf("data/sample.pdf")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([text])
print(f"LangChain split the document into {len(docs)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_documents(docs, embeddings, collection_name="langchain_demo")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("why did the availability filter test fail?")

print(f"\nTop {len(results)} retrieved chunks:\n")
for r in results:
    print(r.page_content)
    print("---")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
prompt = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {question}"
)
context = "\n\n".join([r.page_content for r in results])
chain = prompt | llm
response = chain.invoke({"context": context, "question": "why did the availability filter test fail?"})
print(f"\nAnswer: {response.content}")