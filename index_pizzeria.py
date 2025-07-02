import pandas as pd
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import chromadb

COLLECTION_NAME = "pizzeria_rag_collection"
EMBEDDING_MODEL = "mxbai-embed-large"

# Lecture des CSV
menu_df = pd.read_csv('Menu_Vapiano.csv')
allergen_list_df = pd.read_csv('Liste_allergenes.csv')
allergens_per_dish_df = pd.read_csv('Dishes_Allergens.csv')

# Mapping allergènes
allergen_map = dict(zip(allergen_list_df['Code'].astype(str), allergen_list_df['Allergène']))

# Création des documents
documents = []

for idx, row in menu_df.iterrows():
    dish_name = row['Plat']
    price = row['Prix (€)']
    ingredients = row['Ingrédients']
    category = row['Catégorie']

    allergens_row = allergens_per_dish_df[allergens_per_dish_df['Dish'] == dish_name]
    if not allergens_row.empty:
        allergen_codes = allergens_row['Allergens'].iloc[0].split(',')
        allergen_labels = [allergen_map.get(code.strip(), f'Code inconnu {code.strip()}') for code in allergen_codes]
        allergen_text = ', '.join(allergen_labels)
    else:
        allergen_text = "Non spécifié"

    text = f"""Nom du plat : {dish_name}
Catégorie : {category}
Prix : {price} €
Ingrédients : {ingredients}
Allergènes : {allergen_text}
"""
    documents.append(Document(page_content=text))

# Embedding + Vectorisation
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=ollama_embeddings,
    collection_name=COLLECTION_NAME,
    client=chromadb.PersistentClient(path="./chroma_db2")
)

print(f"✅ {len(documents)} documents indexés avec succès dans ChromaDB.")
