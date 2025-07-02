import os

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

import json

import chromadb
import requests
from chromadb.utils import embedding_functions

# --- 1. Configuration ---

# Le nom de la collection que nous avons créée dans le script précédent.
COLLECTION_NAME = "cours_rag_collection"
# Le modèle d'embedding (doit être le même que celui utilisé pour la création).
EMBEDDING_MODEL = "mxbai-embed-large"
# Le modèle de LLM à utiliser pour la génération de la réponse.
LLM_MODEL = "mistral"

# --- 2. Initialisation des clients ---

print("Initialisation des clients...")
# Initialise le client ChromaDB pour se connecter à la base de données existante.
client = chromadb.PersistentClient(path="./chroma_db")

# Initialise la fonction d'embedding via Ollama.
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

# Charge la collection existante.
# Note: Si la collection n'existe pas, une erreur sera levée.
# Assurez-vous d'avoir exécuté le script `module5_creation_db.py` avant.
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ollama_ef)

# --- 3. Le processus RAG ---


def rag(question):
    """
    Exécute le processus RAG complet : recherche, augmentation, génération.
    """
    # 1. ÉTAPE DE RECHERCHE (Retrieval)
    # Interroge la collection ChromaDB pour trouver les 3 chunks les plus pertinents.
    results = collection.query(query_texts=[question], n_results=3)
    print(results)
    # Récupère le contenu des documents (chunks) trouvés.
    context = "\n".join(results["documents"][0])

    # 2. ÉTAPE D'AUGMENTATION (Augmented)
    # Crée le template du prompt.
    prompt_template = (
        """Tu as 2 fichiers. Dans le Menu.pdf tu trouveras des listes d'ingrédients de plats Pasta, Pizza, Risotto, Antipasti, Dolci et Insalata.
        Quand on te demande une recette, ou une liste d'ingrédients, n'utilise que des chunks du fichier Menu.pdf .
        Le prix est toujours spécifié après le nom de la recette. Les listes d'ingrédients sont toujours spécifiés après les prix.  
        Par exemple, le prix de la recette Quattro Formaggi est 14.10€.
        La recette de la Pizza Quattro Formaggi contient "Mozzarella, gorgonzola, chèvre, burrata, roquette,noisettes torrefiées et crème.

        --- Contexte ---\n
        {context}"
        --- Fin du Contexte ---\n\n
        Question: {question}"""
    )
    # Remplit le template avec le contexte et la question.
    prompt = prompt_template.format(context=context, question=question)

    # 3. ÉTAPE DE GÉNÉRATION (Generation)
    # Prépare les données pour l'API de chat d'Ollama.
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    # Envoie la requête au LLM.
    response = requests.post("http://localhost:11434/api/chat", json=data)

    # Retourne la réponse du LLM.
    if response.status_code == 200:
        return json.loads(response.text)["message"]["content"]
    else:
        return f"Erreur lors de la génération : {response.text}"


# --- 4. Boucle d'interaction ---

if __name__ == "__main__":
    print("\n--- Chatbot RAG ---")
    print("Posez des questions sur le document. Tapez 'exit' pour quitter.")

    while True:
        user_question = input("\nVous: ")
        if user_question.lower() == "exit":
            break

        print("Assistant: ...")
        answer = rag(user_question)
        print(f"\rAssistant: {answer}")
