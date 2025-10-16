#!/usr/bin/env python3
"""
Test script for the new document-based processing functionality.
"""

import requests
import json

def test_document_processing():
    """Test the new document processing endpoint."""
    
    # Test content in English (as per your example)
    test_content = "I am embarrassed. I went to school. I am happy today. My brother is tall. The hotel is near. The university is far. I am a student. I bought a newspaper. I work in an office."
    
    # API endpoint
    url = "http://localhost:8000/familiarity"
    
    # Request payload matching your exact specification
    payload = {
        "learning_language": "eng",
        "native_language": "spa", 
        "content": test_content
    }
    
    try:
        print("Testing content familiarity API...")
        print(f"Content: {test_content}")
        print(f"Learning Language: {payload['learning_language']}")
        print(f"Native Language: {payload['native_language']}")
        print("-" * 50)
        
        # Make request
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Success! Processed {result['total_tokens']} tokens in {len(result['sentences'])} sentences")
            print()
            
            # Display results by sentence
            for sentence in result['sentences']:
                print(f"Sentence {sentence['index']}: {sentence['text']}")
                print("  Tokens:")
                for token in sentence['tokens']:
                    cognate_info = ""
                    if 'cognate_familiarity_score' in token:
                        cognate_info = f" (cognate: {token.get('cognate', 'N/A')} -> {token['cognate_familiarity_score']})"
                    print(f"    {token['text']}: {token['familiarity_score']}{cognate_info}")
                print()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_api_info():
    """Test the root endpoint to see API info."""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            info = response.json()
            print("API Info:")
            print(json.dumps(info, indent=2))
        else:
            print(f"❌ Error getting API info: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")

if __name__ == "__main__":
    print("🧪 Testing Content-Based Word Familiarity API")
    print("=" * 60)
    
    test_api_info()
    print()
    test_document_processing()