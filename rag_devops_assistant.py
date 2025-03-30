from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vector store
vector_store = FAISS.load_local("devops_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load the Phi-2 model with 4-bit quantization
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config={"load_in_4bit": True}
)

def get_response(query):
    """Fetches response from the model based on a query."""
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Context: {context}
    Question: {query}
    Answer:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)  # Fixed truncation issue
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return f"""
    💡 **DevOps Assistant Response**
    
    📌 **Issue:** {query}
    
    ✅ **Solution:**
    {response}
    
    ---
    📘 **Need More Help?** Ask a more specific question!
    """

# Example usage
if __name__ == "__main__":
    query = "How can I set up a CI/CD pipeline using GitHub Actions for an Azure-based microservices project?"
    response = get_response(query)
    print(response)
