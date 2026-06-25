<div align="center">

# 📊 InsightSphere

### *Ask. Analyze. Invest.*

**An AI-powered financial research assistant that transforms news articles into actionable investment insights.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-AI-F55036?style=for-the-badge&logo=lightning&logoColor=white)](https://groq.com)

---

</div>

InsightSphere lets you paste up to 3 financial news article URLs, processes them using AI, and lets you ask natural language questions — getting precise, source-grounded answers in seconds. Built for equity analysts and investors who want to cut through the noise fast.

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **Secure Auth** | User registration & login with bcrypt-hashed passwords stored in SQLite |
| 🌐 **Smart Scraping** | Fetches and cleans article content from any public financial news URL |
| 🤖 **AI Summarization** | Extracts key financial data — figures, forecasts, and events — from articles instantly |
| 🔍 **Semantic Search** | FAISS vector store + HuggingFace embeddings for context-aware Q&A |
| 💬 **RAG Q&A** | Ask natural language questions — get precise, source-grounded answers |
| 🎨 **Modern UI** | Dark glassmorphism interface with animated Siri-style orb, micro-animations, and smooth transitions |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | Groq API |
| **Embeddings** | HuggingFace Sentence Transformers |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **RAG Framework** | LangChain |
| **HTML Parsing** | BeautifulSoup4, readability-lxml, html2text |
| **Auth** | SQLite + bcrypt |
| **Config** | python-dotenv |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/insightsphere.git
cd insightsphere
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📖 How to Use

1. **Register / Login** — Create an account or sign in with existing credentials
2. **Paste URLs** — Add up to 3 financial news article URLs in the sidebar (Bloomberg, Reuters, WSJ, Moneycontrol, etc.)
3. **Process Articles** — Click **⚡ Process Articles** and wait for AI summarization to complete
4. **Ask Questions** — Type any financial question in the input box:
   - *"What were Apple's Q2 earnings and analyst reactions?"*
   - *"What is the revenue growth forecast for this company?"*
   - *"How did the market react to this announcement?"*
5. **Get Insights** — Receive precise, source-backed answers in seconds

---

<div align="center">
Made with ❤️ for equity analysts and investors
</div>
