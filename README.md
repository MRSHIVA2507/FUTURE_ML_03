# AI Resume Screening & Ranking System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)

A Machine Learning-based resume screening system designed to help HR teams and recruiters automate the initial screening process. This tool reads resumes, extracts skills, compares them against a job description, and ranks candidates based on relevance.

> **Why this matters:** Hiring teams often receive hundreds of resumes for a single role. Manually reviewing them is slow and error-prone. This system uses NLP to shortlist candidates faster, identify skill gaps, and reduce recruiter workload.

## 🎯 Objective
To build an intelligent system that can:
- **Read** unstructured resume text (PDF/DOCX/TXT).
- **Extract** key skills and relevant keywords using NLP (SpaCy).
- **Rank** candidates based on semantic similarity to the Job Description.
- **Visualize** the "Skill Gap" (what matches vs. what's missing).

## ✨ Key Features
- **📄 Multi-Format Support**: Upload multiple PDF, DOCX, or TXT resumes at once.
- **🧠 Semantic Matching**: Uses TF-IDF vectorization and Cosine Similarity to understand context, not just keywords.
- **🔍 Skill Gap Analysis**: clearly highlights **✅ Matched Skills** and **❌ Missing Skills** for every candidate.
- **📊 Resume Analysis**: Visualizes the distribution of your candidate pool (e.g., "How many are Java Developers vs Python Developers?").
- **🤖 JD Helper**: Smartly suggests requirements for your Job Description based on top skills found in your dataset.

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (Interactive Web UI)
- **NLP**: [NLTK](https://www.nltk.org/) (Lemmatization, Tokenization, Stopwords)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (TF-IDF, Cosine Similarity)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [PDFPlumber](https://github.com/jsvine/pdfplumber)

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher installed.

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/resume-ranking-system.git
    cd resume-ranking-system
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will also download the necessary SpaCy model (`en_core_web_sm`).*

3.  **Run the Application**
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **Access the Dashboard**
    Open your browser at `http://localhost:8501`.

## 📖 How it Works

1.  **Upload Resumes**: Go to the **Rank Resumes** tab and upload your candidate files.
2.  **Input Job Description**: Paste the JD text. You can also use the **JD Helper** tab to generate one.
3.  **Rank**: Click the button. The system cleans the text, vectorizes it, and calculates a similarity score (0-100%).
4.  **Review**: Expand the candidate cards to see a detailed breakdown of matched and missing skills.

## 📂 Project Structure
```
├── data/                   # Dataset folder (Resume.csv for analysis)
├── streamlit_app.py        # Main Application Interface
├── utils.py                # Core NLP & Processing Logic
├── requirements.txt        # Project Dependencies
└── README.md               # Documentation
```

## 🔮 Future Scope
- **Neural Networks**: Implementing BERT/RoBERTa for deeper semantic understanding.
- **Resume Parsing API**: Extracting Name, Email, and Phone Number structured data.
- **Feedback Loop**: Allowing recruiters to manually "Like/Dislike" a ranking to retrain the model.

---
*Built for the "AI for HR" implementation task.*
