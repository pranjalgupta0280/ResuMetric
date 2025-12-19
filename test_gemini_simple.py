# test_gemini_fixed.py
import google.generativeai as genai
import os

print("="*60)
print("GEMINI API DIAGNOSTIC TEST - FIXED VERSION")
print("="*60)

# Your API key
YOUR_KEY = "AIzaSyDlEU79NOj-OIc3sEG1ahSJVhBwLDWYh8g"

print(f"\n1. Testing key: {YOUR_KEY[:10]}...")
print(f"   Key length: {len(YOUR_KEY)} characters")

try:
    # Test 1: Basic configuration
    print("\n2. Configuring Gemini...")
    genai.configure(api_key=YOUR_KEY)
    print("   ✅ Configuration successful")
    
    # Test 2: LIST AVAILABLE MODELS FIRST
    print("\n3. Listing available models...")
    try:
        models = genai.list_models()
        print("   Available models:")
        for model in models:
            print(f"   - {model.name}")
    except:
        print("   Could not list models (older API version)")
    
    # Test 3: Try CORRECT model names
    print("\n4. Testing with CORRECT model names...")
    
    # CORRECT model names for your API version
    correct_models = [
        'gemini-1.0-pro',           # Most common
        'models/gemini-1.0-pro',    # Full path
        'gemini-pro',               # Try without version
        'tunedModels/gemini-1.0-pro-001',  # Tuned model
    ]
    
    for model_name in correct_models:
        print(f"   Testing: {model_name}")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello World'")
            
            if response and hasattr(response, 'text'):
                print(f"   ✅ SUCCESS! Response: '{response.text}'")
                print(f"\n🎉 Your Gemini API is WORKING with model: {model_name}")
                print(f"   Use this model in your app.py")
                break
            else:
                print(f"   ❌ No response text")
                
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print(f"   ❌ Model not found (404)")
            elif "permission" in error_msg.lower():
                print(f"   ❌ Permission denied")
            else:
                print(f"   ❌ Error: {error_msg[:60]}")
            continue
    
    # If none worked, try the DEFAULT model
    print("\n5. Trying default model...")
    try:
        model = genai.GenerativeModel('gemini-pro')  # Default
        response = model.generate_content("Hello")
        if response.text:
            print(f"   ✅ Default model works! Response: '{response.text}'")
    except Exception as e:
        print(f"   ❌ Default failed: {str(e)[:60]}")
            
except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {type(e).__name__}: {str(e)}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)