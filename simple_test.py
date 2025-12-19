# simple_test.py
import sys
print("Python version:", sys.version)
print("Python path:", sys.executable)

try:
    import streamlit
    print("✅ Streamlit version:", streamlit.__version__)
except ImportError:
    print("❌ Streamlit not installed")

try:
    import google.generativeai
    print("✅ google-generativeai installed")
except ImportError:
    print("❌ google-generativeai not installed")