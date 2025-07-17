import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
from langdetect import detect
import re

# --- Page setup (must be first Streamlit command) ---
st.set_page_config(page_title="🌏 Translator & Company Chooser", layout="centered")

# --- Session state defaults ---
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

if 'name' not in st.session_state:
    st.session_state.name = ""

if 'sector' not in st.session_state:
    st.session_state.sector = ""

if 'company' not in st.session_state:
    st.session_state.company = ""

# --- Load translation models ---
@st.cache_resource
def load_models():
    en_zh_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    en_zh_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    zh_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    zh_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return en_zh_tokenizer, en_zh_model, zh_en_tokenizer, zh_en_model

en_zh_tokenizer, en_zh_model, zh_en_tokenizer, zh_en_model = load_models()

# --- Translation logic ---
def is_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def is_english(text):
    english_chars = sum(c.isascii() and c.isalpha() for c in text)
    return english_chars / max(len(text), 1) > 0.6

def translate(text: str):
    try:
        lang = detect(text)
    except:
        return "Could not detect language.", ""

    if is_chinese(text) or lang.startswith("zh"):
        tokenizer = zh_en_tokenizer
        model = zh_en_model
        direction = "ZH → EN"
    elif is_english(text) or lang == "en":
        tokenizer = en_zh_tokenizer
        model = en_zh_model
        direction = "EN → ZH"
    else:
        return f"Unsupported language: {lang}", ""

    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    return result, direction

def simple_bot_reply(translated_text):
    if "hello" in translated_text.lower():
        return "Hi there! How can I help you today?"
    elif "meeting" in translated_text.lower():
        return "We’ll schedule the meeting shortly!"
    elif "thank" in translated_text.lower():
        return "You're very welcome!"
    else:
        return "Thanks for your message! We'll respond soon."

# --- Company Selector UI ---
if not st.session_state.setup_complete:
    st.title("🏢 Company Collaboration Chooser")

    st.session_state.name = st.text_input("What is your name?", value=st.session_state.name)

    dic = {
        "Games": ["Nintendo", "Riot Games", "Netease"],
        "Cars": ["Mercedes", "Lamborghini", "Rolls Royce"],
        "Technology": ["Logitech", "Pulsar", "Razer"],
        "Watches": ["Rolex", "Omega", "Grand Seiko"]
    }

    st.session_state.sector = st.selectbox("Select a sector:", list(dic.keys()), index=0)
    st.session_state.company = st.selectbox(f"Choose a company in {st.session_state.sector}:", dic[st.session_state.sector])

    if st.button("✅ Confirm and Proceed to Translator"):
        if st.session_state.name.strip() == "":
            st.warning("Please enter your name.")
        else:
            st.session_state.setup_complete = True
            st.rerun()

# --- Translator Interface with Back Button ---
if st.session_state.setup_complete:
    st.title("💬 AI Translator Chat")

    st.success(f"Welcome {st.session_state.name}! You're collaborating with **{st.session_state.company}** in the **{st.session_state.sector}** sector.")

    # Back button
    if st.button("🔙 Go Back to Company Selection"):
        st.session_state.setup_complete = False
        st.rerun()

    user_input = st.text_area("Send a message (EN/中文):", height=100)

    if st.button("Translate and Chat"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            translated, direction = translate(user_input)
            if translated:
                st.markdown(f"**Translated ({direction}):** {translated}")
                bot_reply = simple_bot_reply(translated)
                st.markdown(f"**🤖 Bot Reply:** {bot_reply}")
            else:
                st.error("Translation failed.")
