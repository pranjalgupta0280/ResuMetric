import streamlit as st
import google.generativeai as genai

st.title("Gemini Test")

# Manual key input
api_key = st.text_input("Enter Gemini API Key", type="password")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        if st.button("Test Connection"):
            with st.spinner("Testing..."):
                response = model.generate_content("Say 'Hello World'")
                st.success("✅ Connected!")
                st.write("Response:", response.text)
                
                # Save to secrets
                import os
                os.makedirs(".streamlit", exist_ok=True)
                with open(".streamlit/secrets.toml", "w") as f:
                    f.write(f'GEMINI_API_KEY = "{api_key}"')
                st.info("Key saved to secrets.toml")
                
    except Exception as e:
        st.error(f"❌ Error: {e}")