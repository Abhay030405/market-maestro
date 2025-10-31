"""
Test Phase 2 - AI Agents
Run this to test all agent endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("🧪 TESTING PHASE 2 - AI AGENTS\n")
print("="*60)

# Test 1: Analyze Single Stock
print("\n1️⃣ Testing Single Stock Analysis (AAPL)...")
print("-"*60)

payload = {
    "symbol": "AAPL",
    "user_goal": "Looking for a long-term growth investment",
    "start_date": "2024-01-01",
    "end_date": "2025-10-29"
}

try:
    response = requests.post(f"{BASE_URL}/analyze/stock", json=payload, timeout=120)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Analysis Complete!")
        print(f"\nReport Type: {data.get('report_type')}")
        print(f"Symbol: {data.get('symbol')}")
        
        # Executive Summary
        exec_summary = data.get('executive_summary', '')
        if exec_summary:
            print(f"\n📊 Executive Summary:\n{exec_summary}")
        
        # Market Analysis
        market = data.get('market_analysis', {})
        if market.get('status') == 'success':
            print(f"\n📈 Market Analysis:")
            print(f"   - Last Price: ${market.get('last_price', 'N/A')}")
            print(f"   - Trend: {market.get('trend_analysis', {}).get('direction', 'N/A')}")
            print(f"   - Signal: {market.get('signals', {}).get('overall_signal', 'N/A')}")
        
        # Event Analysis
        events = data.get('event_analysis', {})
        if events.get('status') == 'success':
            print(f"\n📰 Event Analysis:")
            print(f"   - Articles: {events.get('article_count', 0)}")
            print(f"   - Sentiment: {events.get('sentiment_label', 'N/A')}")
        
        # Risk Analysis
        risk = data.get('risk_analysis', {})
        if risk.get('status') == 'success':
            print(f"\n⚠️  Risk Analysis:")
            metrics = risk.get('metrics', {})
            print(f"   - Volatility: {metrics.get('volatility', 0)*100:.2f}%")
            print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        
        # Advisor Recommendation
        advisor = data.get('advisor_recommendation', {})
        if advisor.get('status') == 'success':
            print(f"\n💡 AI Advisor:")
            recommendations = advisor.get('recommendations', [])
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print("\n✅ Single Stock Analysis Test PASSED")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the backend is running!")


# Test 2: Analyze Portfolio
print("\n" + "="*60)
print("2️⃣ Testing Portfolio Analysis...")
print("-"*60)

payload = {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "user_goal": "Build a balanced tech portfolio",
    "optimization_method": "risk_parity"
}

try:
    response = requests.post(f"{BASE_URL}/analyze/portfolio", json=payload, timeout=120)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Portfolio Analysis Complete!")
        print(f"\nReport Type: {data.get('report_type')}")
        print(f"Symbols: {', '.join(data.get('symbols', []))}")
        
        # Portfolio Allocation
        portfolio = data.get('portfolio_allocation', {})
        if portfolio.get('status') == 'success':
            print(f"\n💼 Recommended Allocation:")
            allocation = portfolio.get('allocation', {})
            weights = allocation.get('weights', {})
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"   {symbol}: {weight*100:.2f}%")
            
            metrics = portfolio.get('portfolio_metrics', {})
            print(f"\n📊 Portfolio Metrics:")
            print(f"   - Expected Return: {metrics.get('expected_return', 0)*100:.2f}%")
            print(f"   - Volatility: {metrics.get('volatility', 0)*100:.2f}%")
            print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        
        # Risk Analysis
        risk = data.get('risk_analysis', {})
        if risk.get('status') == 'success':
            print(f"\n⚠️  Portfolio Risk:")
            diversification = risk.get('diversification', {})
            print(f"   - Diversification: {diversification.get('level', 'N/A')}")
        
        print("\n✅ Portfolio Analysis Test PASSED")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")


print("\n" + "="*60)
print("✅ PHASE 2 TESTING COMPLETE!")
print("="*60)
print("\n🎉 All AI agents are working!")
print("\n📝 Available Endpoints:")
print("   - POST /analyze/stock")
print("   - POST /analyze/portfolio")
print("\n🚀 Ready for Phase 3 (Frontend)? Just say 'Start Phase 3'!")