import spacy
import wikipedia
from transformers import pipeline

# Load NLP models
nlp = spacy.load("en_core_web_sm")
# nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

def extract_claims(text):
    """Break text into sentences and extract potential factual claims."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def retrieve_fact(query):
    """Fetch summary from Wikipedia for claim verification."""
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return None

def verify_claim(claim, fact):
    """Check if claim is supported by the retrieved fact using NLI."""
    if not fact:
        return "Not Verifiable ⚠️"

    # Pass (premise=fact, hypothesis=claim)
    result = nli_model(fact, text_pair=claim, truncation=True)[0]
    label = result['label']

    if label == "ENTAILMENT":
        return "Supported ✅"
    elif label == "CONTRADICTION":
        return "Contradicted ❌"
    else:
        return "Not Verifiable ⚠️"

def check_text(text):
    claims = extract_claims(text)
    results = []
    for claim in claims:
        fact = retrieve_fact(claim)
        status = verify_claim(claim, fact)
        results.append((claim, status, fact))
    return results

