import os
from groq import Groq
from dotenv import load_dotenv

# Load your existing API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

print("🔍 Fetching your active Groq models...")
models = client.models.list()

print("\n✅ ACTIVE MODELS FOR YOUR API KEY:")
print("-" * 40)
for m in models.data:
    print(m.id)
print("-" * 40)