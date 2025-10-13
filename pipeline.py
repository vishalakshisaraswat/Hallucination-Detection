import spacy
import wikipedia
from transformers import pipeline

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load NLI model for factual verification
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")


def extract_claims(text):
    """Break text into sentences and extract potential factual claims."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def retrieve_fact(claim):
    """Fetch relevant fact from Wikipedia using named entities or main nouns."""
    doc = nlp(claim)
    # Try named entities first (person, place, org, etc.)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]]

    # Fallback: use main noun chunk if no entity found
    if not entities:
        entities = [chunk.text for chunk in doc.noun_chunks][:1]

    # Try fetching Wikipedia summary for each candidate
    for ent in entities:
        try:
            summary = wikipedia.summary(ent, sentences=2, auto_suggest=False)
            return summary
        except Exception:
            continue
    return None


def verify_claim(claim, fact):
    """Check if claim is supported, contradicted, or unverifiable."""
    if not fact:
        return "Not Verifiable ⚠️"

    # Run NLI model (premise=fact, hypothesis=claim)
    result = nli_model(fact, text_pair=claim, truncation=True)[0]
    label = result["label"].upper()

    if "ENTAILMENT" in label:
        return "TRUE ✅"
    elif "CONTRADICTION" in label:
        return "FALSE ❌"
    else:
        return "Not Verifiable ⚠️"


def check_text(text):
    """Main pipeline to process text and return verification results."""
    claims = extract_claims(text)
    results = []

    for claim in claims:
        fact = retrieve_fact(claim)
        status = verify_claim(claim, fact)
        results.append((claim, status, fact))

    return results