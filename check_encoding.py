# check_encoding.py
import os

secrets_path = ".streamlit/secrets.toml"

if not os.path.exists(secrets_path):
    print("❌ File doesn't exist")
else:
    # Try different encodings
    encodings = ['utf-8', 'utf-16', 'latin-1', 'ascii']
    
    for encoding in encodings:
        try:
            with open(secrets_path, 'r', encoding=encoding) as f:
                content = f.read()
                print(f"✅ Can read with {encoding:10}: {content[:50]}")
        except UnicodeDecodeError:
            print(f"❌ Failed with {encoding:10}")