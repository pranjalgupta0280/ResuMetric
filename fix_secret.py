# fix_secrets.py
import os

# Create .streamlit folder if it doesn't exist
os.makedirs(".streamlit", exist_ok=True)

# Get your API key
api_key = input("Paste your Gemini API key: ").strip()

# Clean the key - remove any weird characters
import re
api_key = re.sub(r'[^\x20-\x7E]', '', api_key)  # Remove non-ASCII
api_key = api_key.strip()

# Write with proper UTF-8 encoding
with open(".streamlit/secrets.toml", "w", encoding="utf-8") as f:
    f.write(f'GEMINI_API_KEY = "{api_key}"')

print(f"\n✅ Created .streamlit/secrets.toml with key (length: {len(api_key)})")
print(f"🔑 Key starts with: {api_key[:10]}...")

# Verify
with open(".streamlit/secrets.toml", "r", encoding="utf-8") as f:
    print(f"📄 File content: {f.read()}")