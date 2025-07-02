# -*- coding: utf-8 -*-

import chromadb
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 1. Configuration ---

# Le nom de la collection que nous avons créée dans le script précédent.
COLLECTION_NAME = "cours_rag_collection"
# Le modèle d'embedding (doit être le même que celui utilisé pour la création).
EMBEDDING_MODEL = "mxbai-embed-large"
# Le modèle de LLM à utiliser pour la génération de la réponse.
LLM_MODEL = "mistral"

# --- 2. Initialisation des composants LangChain ---

print("Initialisation des composants LangChain...")

# Initialise le client Ollama pour les embeddings
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialise le client ChromaDB pour se connecter à la base de données existante.
# Le chemin doit correspondre à l'endroit où la DB a été créée par module5_creation_db.py
# (qui est dans le même dossier 'code')
vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

# Crée un retriever à partir du vectorstore.
# Le retriever est responsable de la recherche des documents pertinents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Récupère les 3 chunks les plus pertinents

print('#######')
print(retriever)

# Initialise le modèle de chat Ollama
llm = ChatOllama(model=LLM_MODEL)

# --- 3. Définition du prompt RAG ---

# Le template du prompt pour le LLM.
# Il inclut le contexte récupéré et la question de l'utilisateur.
template = """Tu as 2 fichiers. Dans le fichier Menu.pdf tu trouveras des listes d'ingrédients de plats Pasta, Pizza, Risotto, Antipasti, Dolci et Insalata.
Quand on te demande une recette, ou une liste d'ingrédients, n'utilise que des chunks du fichier Menu.pdf .
Le prix est toujours spécifié après le nom de la recette. Les listes d'ingrédients sont toujours spécifiés après les prix.  
Par exemple, le prix de la recette Macaroni Regressi est 10,50€.
La recette de la Pasta Macaroni Regressi contient Macaroni Tomates cerises, oignons, sauce tomate, mozzarella.

Dans le fichier Liste_allergenes.pdg , tu trouveras la liste des allergènes au départ. Chaque allergène correspond à un numéro:
CÉRÉALES OU GLUTEN 1
CRUSTACÉS 2
ŒUFS 3
POISSONS 4
ARACHIDES 5
SOJA 6
LAIT 7
FRUITS À COQUE 8
(amandes, noisettes, noix, noix de cajou, noix de pécan, noix du Brésil, noix de macadamia,
pistaches, pignons de pin)
CÉLERI 9
MOUTARDE 10
GRAINE DE SÉSAME 11
ANHYDRIDES SULFUREUX SULFITES 12
MOLLUSQUES 13
LUPIN 14

chaque recette est ensuite listée et les chiffres en bout de ligne correspondent aux allergènes présents dans la recette

En te basant uniquement sur le context
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. Construction de la chaîne RAG avec LangChain Expression Language (LCEL) ---

# La chaîne RAG est construite en utilisant LCEL pour une meilleure lisibilité et modularité.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # Étape de recherche (Retrieval)
    | prompt                                                  # Étape d'augmentation (Augmented)
    | llm                                                     # Étape de génération (Generation)
    | StrOutputParser()                                       # Parse la sortie du LLM en chaîne de caractères
)

# --- 5. Boucle d'interaction ---

if __name__ == "__main__":
    print("\n--- Chatbot RAG avec LangChain ---")
    print("Posez des questions sur le document. Tapez 'exit' pour quitter.")

    while True:
        user_question = input("\nVous: ")
        if user_question.lower() == "exit":
            break

        print("Assistant: ...")
        # Invoque la chaîne RAG avec la question de l'utilisateur
        answer = rag_chain.invoke(user_question)
        print(f"\rAssistant: {answer}")