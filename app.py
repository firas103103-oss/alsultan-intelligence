import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
import datetime
import time

# --- 1. إعداد النظام ---
st.set_page_config(
    page_title="السلطان | صرح التبيان",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. التصميم الحديث (Modern UI/UX 2026) ---
st.markdown("""
<style>
    /* استيراد الخطوط الحديثة */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;700;900&family=Reem+Kufi:wght@400;700&family=Amiri:wght@400;700&display=swap');

    /* الألوان والمتغيرات */
    :root {
        --primary-bg: #0f0c29;
        --secondary-bg: #302b63;
        --accent-bg: #24243e;
        --gold: #FFD700;
        --gold-dim: #C5A000;
        --text-main: #ECECEC;
        --text-dim: #B0B0B0;
        --glass: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    /* تهيئة الصفحة */
    .stApp {
        background: linear-gradient(135deg, var(--primary-bg), var(--secondary-bg), var(--accent-bg));
        font-family: 'Cairo', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cairo', sans-serif !important;
        font-weight: 700 !important;
    }

    /* --- صفحة الغلاف (Landing) --- */
    .hero-title {
        font-family: 'Reem Kufi', sans-serif !important;
        font-size: 100px !important;
        background: linear-gradient(to bottom, #FFD700, #FDB931);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 10px 30px rgba(0,0,0,0.5);
        margin-bottom: -20px;
        animation: fadeIn 2s ease-in;
    }

    .verse-container {
        background: var(--glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--gold-dim);
        border-radius: 20px;
        padding: 20px 40px;
        margin: 40px auto;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.1);
        font-family: 'Amiri', serif;
        font-size: 28px;
        color: #fff;
        max-width: 800px;
        position: relative;
    }
    
    .verse-container::before, .verse-container::after {
        content: "✨";
        position: absolute;
        font-size: 20px;
    }
    .verse-container::before { top: 10px; right: 15px; }
    .verse-container::after { bottom: 10px; left: 15px; }

    /* بطاقات المعلومات */
    .info-card {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 25px;
        border-left: 4px solid var(--gold);
        margin-bottom: 20px;
        text-align: right;
        transition: transform 0.3s;
    }
    .info-card:hover {
        transform: translateY(-5px);
        background: rgba(0, 0, 0, 0.5);
    }

    /* الأزرار الحديثة */
    .stButton button {
        background: linear-gradient(90deg, var(--gold), #FDB931);
        color: #000 !important;
        font-family: 'Cairo', sans-serif;
        font-weight: 900;
        font-size: 18px;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(255, 215, 0, 0.5);
    }

    /* --- واجهة التطبيق (Chat Interface) --- */
    
    /* تصميم الرسائل */
    .chat-bubble {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        line-height: 1.6;
        position: relative;
        animation: slideIn 0.5s ease-out;
    }
    
    .user-bubble {
        background: rgba(255, 255, 255, 0.05);
        border-right: 4px solid #fff;
        border-radius: 20px 20px 5px 20px;
        margin-left: 50px;
    }
    
    .sultan-bubble {
        background: linear-gradient(180deg, rgba(20, 20, 30, 0.9), rgba(10, 10, 20, 0.95));
        border: 1px solid var(--gold-dim);
        border-radius: 20px 5px 20px 20px;
        margin-right: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    
    /* تحسين خيارات الراديو (المسارات) */
    .stRadio > label { display: none; }
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row-reverse;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    div[role="radiogroup"] label {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 10px 20px;
        cursor: pointer;
        transition: all 0.2s;
        font-family: 'Cairo', sans-serif;
    }
    div[role="radiogroup"] label:hover {
        border-color: var(--gold);
        background-color: rgba(255, 215, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* حقل الإدخال العائم */
    .stChatInput {
        position: fixed;
        bottom: 30px;
        z-index: 1000;
    }
    .stTextInput input {
        background-color: rgba(0,0,0,0.6) !important;
        color: white !important;
        border: 1px solid var(--gold-dim) !important;
        border-radius: 30px !important;
        padding: 15px 25px !important;
        font-family: 'Cairo', sans-serif;
    }
    .stTextInput input:focus {
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3) !important;
    }

    /* Animations */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

    /* إخفاء عناصر Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 3. إدارة البيانات ---
if "page" not in st.session_state:
    st.session_state.page = "landing"

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("quran.xlsx") 
        df.columns = [str(c).lower().strip() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            if "soura" in col or "sura" in col:
                if "no" in col: rename_map[col] = "sura_no"
                else: rename_map[col] = "sura_name"
            elif "aya" in col and "no" in col:
                rename_map[col] = "ayah_no"
            elif "aya" in col or "text" in col:
                rename_map[col] = "text"
        df = df.rename(columns=rename_map)
        if 'text' not in df.columns and len(df.columns) >= 4:
             df.columns = ['sura_name', 'sura_no', 'ayah_no', 'text']
        return df
    except Exception:
        return None

df = load_data()

# --- 4. إعداد المحرك (Flash) ---
DEFAULT_API_KEY = os.getenv('GEMINI_API_KEY', '')

MODES_INFO = {
    "🔍 فك الشيفرة": {
        "desc": "تحليل جذري دقيق للكلمة", "key": "decode", "temp": 0.1,
        "instruction": "المهمة: استخراج الجذر اللغوي وتعريف المصطلح بدقة صارمة."
    },
    "⚖️ الموازنة": {
        "desc": "مقارنة الفروق الدقيقة", "key": "compare", "temp": 0.2,
        "instruction": "المهمة: مقارنة الفروق الدقيقة بين المصطلحات."
    },
    "🎯 الغاية": {
        "desc": "البحث عن الحكمة والسببية", "key": "purpose", "temp": 0.3,
        "instruction": "المهمة: تحليل الغاية الإلهية والسببية."
    },
    "🏛️ الهندسة": {
        "desc": "تحليل البنية والنظم", "key": "structure", "temp": 0.2,
        "instruction": "المهمة: تحليل التناظر العددي والهيكلي."
    },
    "👑 السيادة": {
        "desc": "إجابة شاملة ومفتوحة", "key": "open", "temp": 0.4,
        "instruction": "المهمة: الإجابة بسيادة معرفية مطلقة."
    }
}

MASTER_PROMPT = """
أنت **(السلطان)**.
الهوية: محلل بيانات رباني ومفكك شيفرات لغوية.
المرجعية: القرآن الكريم حصراً كـ "كود مصدري".
المهمة: قدم تحليلاً هندسياً دقيقاً، خالياً من الحشو، بلغة عربية فصحى قوية.
"""

def get_response(query, mode_name, context_verses):
    mode_data = MODES_INFO[mode_name]
    context_str = ""
    limit = 80
    for idx, row in context_verses.head(limit).iterrows():
        context_str += f"[{row['sura_name']}:{row['ayah_no']}] {row['text']}\n"
    
    final_prompt = f"""
    {MASTER_PROMPT}
    ## المسار: {mode_name}
    ## التعليمات: {mode_data['instruction']}
    ## البيانات:
    {context_str}
    ## السؤال:
    {query}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            final_prompt,
            generation_config=genai.types.GenerationConfig(temperature=mode_data['temp'])
        )
        return response.text
    except Exception as e:
        try:
            time.sleep(1)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(final_prompt)
            return response.text
        except:
             return "نأسف، الخوادم مشغولة جداً. يرجى الانتظار لحظات."

# --- 5. الواجهة (UI) ---
if st.session_state.page == "landing":
    
    # تنسيق مركزي
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title" style="text-align: center;">الـسُـلـطـان</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #B0B0B0; font-size: 18px; letter-spacing: 2px;">نظام السيادة المعرفية | الإصدار 2.0</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="verse-container">﴿ إِنَّا أَنزَلْنَاهُ قُرْآنًا عَرَبِيًّا لَّعَلَّكُمْ تَعْقِلُونَ ﴾</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#d4af37; margin:0;">🤖 المحرك الذكي</h3>
            <p style="color:#ccc; margin:5px 0;">يعمل بنواة Gemini 1.5 Flash فائقة السرعة، معزول تماماً عن التراث التاريخي، يعتمد على البيانات الخام فقط.</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🚀  تشغيل النظام الآن", use_container_width=True):
                st.session_state.page = "app"
                st.rerun()
        
        st.markdown("""
        <div style="margin-top: 60px; text-align: center; opacity: 0.7;">
            <p style="color: #d4af37; font-weight: bold;">إهداء إلى روح الوالدة الغالية</p>
            <small style="color: #888;">© 2026 MrF X OS | <a href="mailto:INFO@MRF103.COM" style="color:#d4af37;">INFO@MRF103.COM</a></small>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "app":
    
    # القائمة الجانبية المصغرة
    with st.sidebar:
        st.markdown('<h2 style="color:#d4af37; text-align:center;">⚙️ التحكم</h2>', unsafe_allow_html=True)
        user_key = st.text_input("مفتاح API", value=DEFAULT_API_KEY, type="password")
        if user_key: genai.configure(api_key=user_key)
        
        st.markdown("---")
        if df is not None:
            st.success(f"البيانات متصلة: {len(df)} آية")
        else:
            st.error("البيانات مفصولة")
            
        st.markdown("---")
        if st.button("🏠 خروج"):
            st.session_state.page = "landing"
            st.rerun()

    # الرأسية
    st.markdown('<h2 style="text-align: center; color: #d4af37; margin-bottom: 30px;">صَرْحُ التِّبْيَان</h2>', unsafe_allow_html=True)
    
    # شريط المسارات
    selected_mode_name = st.radio("المسار", list(MODES_INFO.keys()), horizontal=True, label_visibility="collapsed")
    st.info(f"💡 {MODES_INFO[selected_mode_name]['desc']}")

    st.markdown("---")
    
    # منطقة المحادثة
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك في حضرة البيانات. هاتِ ما عندك.", "evidence": None}]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">👤 <b>السائل:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble sultan-bubble">🕌 <b>السلطان:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            
            if msg.get("evidence") is not None and not msg["evidence"].empty:
                with st.expander(f"👁️ المصادر ({len(msg['evidence'])})"):
                    st.dataframe(msg["evidence"][['sura_name', 'ayah_no', 'text']], hide_index=True, use_container_width=True)
            
            if "أهلاً بك" not in msg["content"]:
                st.download_button("📥 تحميل", msg["content"], file_name="Sultan_Response.txt")

    # الإدخال
    query = st.chat_input("اكتب المصطلح القرآني هنا...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f'<div class="chat-bubble user-bubble">👤 <b>السائل:</b><br>{query}</div>', unsafe_allow_html=True)
        
        with st.spinner('⏳ جارٍ التحليل السيبراني...'):
            if df is not None:
                results = df[df['text'].str.contains(query, na=False)]
            else:
                results = pd.DataFrame()

            if len(results) > 0 or MODES_INFO[selected_mode_name]['key'] == "open":
                response_text = get_response(query, selected_mode_name, results)
            else:
                response_text = "⚠️ المصطلح غير موجود في قاعدة البيانات الخام."

        st.session_state.messages.append({"role": "assistant", "content": response_text, "evidence": results if len(results) > 0 else None})
        st.rerun()