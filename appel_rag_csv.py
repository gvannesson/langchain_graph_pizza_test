from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import chromadb

import chromadb


# --- Config ---
COLLECTION_NAME = "pizzeria_rag_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"

# --- Initialisation embeddings et vectorstore --
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db2"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print('############')
print(retriever)

# --- Initialisation LLM ---
llm = ChatOllama(model=LLM_MODEL)

# --- Prompt template ---
template = """

En te basant uniquement sur le context
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# --- Construction de la chaîne RAG (LCEL) ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Recherche
    | prompt                                                    # Prompt avec contexte
    | llm                                                       # Génération réponse
    | RunnableLambda(lambda x: str(x))                                         # Extraction du texte simple
)

# --- Boucle interactive ---
if __name__ == "__main__":
    print("\n--- Chatbot RAG LangChain - Pizzeria ---")
    print("Posez vos questions (tapez 'exit' pour quitter).")

    while True:
        user_question = input("\nVous: ")
        if user_question.lower() in ["exit", "quit", "stop"]:
            break

        print("Assistant: ...")
        answer = rag_chain.invoke(user_question)
        print(f"\rAssistant: {answer}")