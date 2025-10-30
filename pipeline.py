import spacy
import wikipedia
from transformers import pipeline

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# NLI model (commonsense-aware)
nli_model = pipeline("text-classification", model="roberta-large-mnli")

# Text-to-text reasoning model (for generating corrections or explanations)
reasoning_model = pipeline("text2text-generation", model="google/flan-t5-large")


def extract_claims(text):
    """Break text into sentences and extract potential factual claims."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def retrieve_fact(claim):
    """Try to get relevant fact from Wikipedia, if applicable."""
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]]
    if not entities:
        entities = [chunk.text for chunk in doc.noun_chunks][:1]

    for ent in entities:
        try:
            summary = wikipedia.summary(ent, sentences=2, auto_suggest=False)
            return summary
        except Exception:
            continue
    return None


def generate_correction(claim):
    """
    Use the reasoning model (FLAN-T5) to generate the most likely true version
    of the claim when a contradiction is detected and no factual data is available.
    """
    prompt = f"The statement '{claim}' seems false. Suggest a more factually correct version."
    result = reasoning_model(prompt, max_new_tokens=50)[0]['generated_text']
    return result.strip()


def verify_claim(claim, fact=None):
    """Combine factual verification + commonsense correction without hardcoding."""
    premise = fact if fact else "Common sense knowledge about the world."
    result = nli_model(premise, text_pair=claim, truncation=True)[0]
    label = result["label"].upper()

    if "ENTAILMENT" in label:
        return "TRUE ✅", fact
    elif "CONTRADICTION" in label:
        if not fact:
            correction = generate_correction(claim)
            return "FALSE ❌", correction
        return "FALSE ❌", fact
    else:
        return "Not Verifiable ⚠️", fact


def check_text(text):
    """Main pipeline to process text and return verification results."""
    claims = extract_claims(text)
    results = []

    for claim in claims:
        fact = retrieve_fact(claim)
        status, fact_or_correction = verify_claim(claim, fact)
        results.append((claim, status, fact_or_correction if fact_or_correction else "—"))

    return results
