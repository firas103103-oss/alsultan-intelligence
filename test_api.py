import os
import google.generativeai as genai

# ضع مفتاحك هنا للتجربة
MY_KEY = os.getenv('GEMINI_API_KEY', '')  # أو مفتاحك الجديد

genai.configure(api_key=MY_KEY)

print("--- جاري فحص الموديلات المتاحة لهذا المفتاح ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ متاح: {m.name}")
except Exception as e:
    print(f"❌ خطأ في المفتاح أو الاتصال: {e}")