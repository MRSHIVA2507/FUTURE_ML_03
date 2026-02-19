import re
import io
import pdfplumber
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text using NLTK:
    1. Lowercase
    2. Remove special characters
    3. Tokenize & Remove Stopwords
    4. Lemmatization (running -> run)
    """
    if not text:
        return ""
    
    # Lowercase and remove special chars
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return " ".join(tokens)

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    """
    Extracts text from a DOCX file using python-docx.
    """
    doc = docx.Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def parse_resume(file):
    """
    Determines the file type and extracts text accordingly.
    Supported formats: .pdf, .docx, .txt
    """
    filename = file.name.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return str(file.read(), "utf-8")
    else:
        return ""

def load_data(file_path):
    """
    Loads the resume dataset from a CSV file.
    """
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def get_category_stats(df):
    """
    Returns the distribution of categories in the dataset.
    """
    if 'Category' in df.columns:
        return df['Category'].value_counts()
    return pd.Series()

def get_top_keywords(df, category, top_n=10):
    """
    Extracts top N keywords for a given category using TF-IDF.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if 'Category' not in df.columns or category not in df['Category'].unique():
        return []
    
    subset = df[df['Category'] == category]
    text_data = subset['Resume_str'].apply(clean_text)
    
    if text_data.empty:
        return []
        
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum tfidf scores for each term across documents
    sum_tfidf = tfidf_matrix.sum(axis=0)
    
    # Create a list of (term, score) tuples
    keywords = [(feature_names[col], sum_tfidf[0, col]) for col in range(sum_tfidf.shape[1])]
    
    # Sort by score descending
    keywords.sort(key=lambda x: x[1], reverse=True)
    
    return [k[0] for k in keywords[:top_n]]
