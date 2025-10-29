"""
Test if .env file is being loaded correctly
Run from project root: python test_config.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("🔍 Debugging .env Configuration\n")

# Check current directory
print(f"📁 Current Directory: {os.getcwd()}")

# Check if .env exists
env_path = Path('.env')
print(f"📄 .env file path: {env_path.absolute()}")
print(f"📄 .env exists: {env_path.exists()}")

if env_path.exists():
    print(f"📄 .env file size: {env_path.stat().st_size} bytes")
    
    # Show first few lines (without revealing full keys)
    print("\n📝 .env file preview:")
    with open(env_path, 'r') as f:
        lines = f.readlines()[:5]
        for i, line in enumerate(lines, 1):
            # Mask API keys for security
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                value = value.strip()
                if len(value) > 10:
                    masked = value[:4] + '*' * (len(value) - 8) + value[-4:]
                else:
                    masked = '*' * len(value)
                print(f"   {i}. {key}={masked}")
            else:
                print(f"   {i}. {line.strip()}")
else:
    print("❌ .env file NOT FOUND!")
    print("\n📝 Creating a template .env file for you...")
    
    template = """# API Keys (Replace with your actual keys)
NEWS_API_KEY=your_newsapi_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional APIs
ALPHA_VANTAGE_KEY=your_alphavantage_key

# App Configuration
CACHE_TTL_SECONDS=300
DEBUG_MODE=True
API_TITLE=Market Maestro API
API_VERSION=1.0.0
RISK_FREE_RATE=0.02
"""
    
    with open('.env', 'w') as f:
        f.write(template)
    
    print("✅ Created .env template!")
    print("📝 Now edit .env and add your API keys")

print("\n" + "="*60)

# Try loading with dotenv
print("\n🔄 Testing dotenv loading...")
load_dotenv()

news_key = os.getenv("NEWS_API_KEY", "")
gemini_key = os.getenv("GOOGLE_API_KEY", "")

print(f"\n📊 Environment Variables Loaded:")
print(f"   NEWS_API_KEY: {'✅ Found' if news_key else '❌ Not found'}")
if news_key:
    print(f"      Length: {len(news_key)} characters")
    print(f"      Preview: {news_key[:4]}...{news_key[-4:] if len(news_key) > 8 else ''}")
    
    # Check if it's still the placeholder
    if news_key == "your_newsapi_key_here":
        print(f"      ⚠️  Still using placeholder! Replace with actual key.")

print(f"\n   GOOGLE_API_KEY: {'✅ Found' if gemini_key else '❌ Not found'}")
if gemini_key:
    print(f"      Length: {len(gemini_key)} characters")
    print(f"      Preview: {gemini_key[:4]}...{gemini_key[-4:] if len(gemini_key) > 8 else ''}")
    
    # Check if it's still the placeholder
    if gemini_key == "your_gemini_api_key_here":
        print(f"      ⚠️  Still using placeholder! Replace with actual key.")

print("\n" + "="*60)

# Test backend config
print("\n🧪 Testing backend.config module...")
try:
    from backend.config import settings
    
    print(f"   Settings loaded: ✅")
    print(f"   NEWS_API_KEY: {'✅ Set' if settings.NEWS_API_KEY else '❌ Empty'}")
    print(f"   GOOGLE_API_KEY: {'✅ Set' if settings.GOOGLE_API_KEY else '❌ Empty'}")
    
    # Validate
    validation = settings.validate_keys()
    print(f"\n   Validation Results:")
    print(f"      news_api: {'✅' if validation['news_api'] else '❌'}")
    print(f"      google_gemini: {'✅' if validation['google_gemini'] else '❌'}")
    
except Exception as e:
    print(f"   ❌ Error importing backend.config: {e}")

print("\n" + "="*60)
print("\n✅ Diagnosis Complete!")
print("\n📋 Next Steps:")
print("   1. If .env doesn't exist → I just created it for you")
print("   2. If keys are placeholders → Replace with real API keys")
print("   3. If keys are set but not detected → Check file encoding (should be UTF-8)")
print("   4. After fixing → Restart the backend server")
print("\n🔗 Get API Keys:")
print("   NewsAPI: https://newsapi.org/register")
print("   Gemini: https://aistudio.google.com/app/apikey")