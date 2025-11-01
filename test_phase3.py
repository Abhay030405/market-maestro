"""
Test Phase 3 - Natural Language Query System
Tests the AI reasoning layer with natural language queries
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("🧪 TESTING PHASE 3 - NATURAL LANGUAGE QUERY SYSTEM\n")
print("="*70)

# Test queries
test_queries = [
    "Analyze Apple stock for long-term investment",
    "Compare TCS and Infosys for short-term growth potential",
    "Optimize my portfolio with Microsoft, Google, and Amazon",
    "What's the risk level of Tesla stock?",
    "Show me the latest news about Nvidia"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}: {query}")
    print("-"*70)
    
    try:
        payload = {"query": query}
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Query Processed Successfully!")
            print(f"\n📊 Detected Intent: {data.get('intent', {}).get('intent_type', 'N/A')}")
            print(f"🎯 Symbols Found: {', '.join(data.get('intent', {}).get('symbols', []))}")
            print(f"⏱️  Time Horizon: {data.get('intent', {}).get('time_horizon', 'Not specified')}")
            
            print(f"\n🤖 AI Response:")
            print("-"*70)
            nl_response = data.get('natural_language_response', '')
            if nl_response:
                # Print in chunks for readability
                words = nl_response.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > 70:
                        print(line)
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    print(line)
            else:
                print("No response generated")
            
            print(f"\n📋 Agents Called: {', '.join(data.get('actions_taken', []))}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text[:200])
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the backend is running!")

print(f"\n{'='*70}")
print("✅ PHASE 3 TESTING COMPLETE!")
print("="*70)
print("\n💡 What we tested:")
print("   - Natural language query parsing")
print("   - Intent detection (analyze, compare, optimize, etc.)")
print("   - Symbol extraction from text")
print("   - Agent routing based on query type")
print("   - Natural language response generation")
print("\n🎯 Next: Test in Swagger UI at http://localhost:8000/docs")
print("   Look for POST /query endpoint")