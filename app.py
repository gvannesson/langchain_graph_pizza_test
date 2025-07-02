import gradio as gr
from rag_langchain_pizza import rag_chain

# --- 5. Fonction de réponse pour Gradio ---
def respond_to_question(question):
    return rag_chain.invoke(question)

# --- 6. Interface Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("## 🍕 Chatbot Polo – Trouvez vos allergènes et recettes Marco Fuso")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(
                placeholder="Posez une question sur les recettes ou les allergènes...",
                label="Votre question"
            )
            submit_btn = gr.Button("Envoyer")

        def handle_user_input(message, history):
            response = respond_to_question(message)
            history = history + [(message, response)]
            return history, ""

        submit_btn.click(handle_user_input, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
        user_input.submit(handle_user_input, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

# Lancer l'interface
demo.launch()