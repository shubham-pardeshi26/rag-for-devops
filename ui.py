import gradio as gr
import requests

def query_assistant(user_input):
    response = requests.post("http://127.0.0.1:8000/query", json={"query": user_input})
    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    return "Error: Unable to reach the assistant."

# Create Gradio UI
iface = gr.Interface(
    fn=query_assistant,
    inputs=gr.Textbox(placeholder="Ask me about DevOps..."),
    outputs="text",
    title="🚀 DevOps Assistant",
    description="Ask any DevOps-related question, and I'll provide answers based on my knowledge base!"
)

# Launch the UI
if __name__ == "__main__":
    iface.launch(share=True)
