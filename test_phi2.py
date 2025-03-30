from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure BitsAndBytes for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Fix for slow inference warning
    bnb_4bit_use_double_quant=True,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Load the Phi-2 model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,  # Use new config format
    device_map="auto",  # Auto-detect GPU
)

print("✅ Phi-2 model loaded successfully!")

# Generate a test response
input_text = "What is the capital of India?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Move input to GPU
output = model.generate(**inputs, max_length=50)

# Decode and print the output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response:", response)
