from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rapidfuzz import fuzz, process
import spacy
import csv

# ----------------------------------------------------------------------------------------------------------------------
# (1) Negation Detection Functions (adapted for direct entity removal)
# ----------------------------------------------------------------------------------------------------------------------
def find_negated_entities_in_sentence(sentence, entities, negation_keywords):
    """
    Given a spaCy Sentence object (sentence) and a list of recognized 'entities',
    return which entities are negated if the sentence contains any negation keywords.
    An entity is considered negated if it appears fully within that sentence (char range).
    """
    sent_lower = sentence.text.lower()

    # If no negation keyword is present, skip negation checks
    if not any(neg_word in sent_lower for neg_word in negation_keywords):
        return []

    sent_start = sentence.start_char
    sent_end = sentence.end_char

    negated = []
    # Check which entities lie fully inside this sentence
    for entity in entities:
        if entity['start'] >= sent_start and entity['end'] <= sent_end:
            negated.append(entity['word'])
    return negated


def remove_negated_entities(ner_results, text, debug=False):
    """
    Using spaCy for sentence segmentation, detect any sentences containing negation 
    keywords. Remove recognized entities that fall within those sentences.
    If debug=True, print the negated entities.
    """

    negation_keywords = {
        "no", "not", "none", "negative", "denies",
        "without", "never", "absent", "does not",
        "did not", "is not", "are not", "No "
    }

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    if debug:
        print("=== Sentences from spaCy (Original Text) ===")
        for sent in doc.sents:
            print(
                f"SENTENCE: {repr(sent.text)} "
                f"| START: {sent.start_char} "
                f"| END: {sent.end_char}"
            )

    negated_entity_indices = set()
    negated_entities_list = []

    for sent in doc.sents:
        negated = find_negated_entities_in_sentence(sent, ner_results, negation_keywords)
        if negated:
            for i, entity in enumerate(ner_results):
                if entity['word'] in negated:
                    negated_entity_indices.add(i)
            negated_entities_list.extend(negated)

    filtered_results = [
        ent for i, ent in enumerate(ner_results)
        if i not in negated_entity_indices
    ]

    if debug:
        print("\n=== Negated Entities ===")
        if negated_entities_list:
            print(", ".join(sorted(set(negated_entities_list))))
        else:
            print("None found")

    return filtered_results

# ----------------------------------------------------------------------------------------------------------------------
# (2) ICD Loading and Matching
# ----------------------------------------------------------------------------------------------------------------------
def load_icd_codes(file_path):
    """
    Loads ICD-10 codes from a text file.
    """
    icd_codes = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)  # split on first whitespace
            if len(parts) == 2:
                code = parts[0].strip()
                description = parts[1].strip()
                icd_codes.append({"code": code, "description": description})
    return icd_codes

def get_best_icd_match(entity_text, icd_codes, score_cutoff=60):
    """
    Fuzzy match a string (entity_text) against the list of ICD descriptions,
    returning the best match above 'score_cutoff'.
    """
    entity_lower = entity_text.lower()
    descriptions = [icd["description"].lower() for icd in icd_codes]

    best_match = process.extractOne(entity_lower, descriptions, scorer=fuzz.WRatio)
    if best_match and best_match[1] >= score_cutoff:
        best_description_lower, best_score, best_index = best_match
        best_code = icd_codes[best_index]["code"]
        best_description = icd_codes[best_index]["description"]
        return best_code, best_description, best_score
    else:
        return None, None, 0

# ----------------------------------------------------------------------------------------------------------------------
# (3) NER Pipeline Helpers
# ----------------------------------------------------------------------------------------------------------------------
def create_biomedical_ner_pipeline(model_name="blaze999/Medical-NER"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner_pipeline

def filter_by_confidence(ner_results, threshold=0.75):
    return [entity for entity in ner_results if entity["score"] >= threshold]

def filter_icd_entities(ner_results):
    """
    Keep only those entity groups that typically map to ICD codes:
    DISEASE_DISORDER, SIGN_SYMPTOM, MEDICATION, etc.
    """
    relevant_groups = {"DISEASE_DISORDER", "SIGN_SYMPTOM"}
    return [ent for ent in ner_results if ent.get("entity_group") in relevant_groups]

def merge_subword_tokens(token_list):
    merged_tokens = []
    buffer = []
    
    for token in token_list:
        if not buffer:
            buffer.append(token)
        else:
            last = buffer[-1]
            same_entity_group = (token["entity_group"] == last["entity_group"])
            is_subword = token["word"].startswith("##")
            adjacent_positions = (token["start"] == last["end"])

            if same_entity_group and (is_subword or adjacent_positions):
                buffer.append(token)
            else:
                merged_tokens.append(_combine_buffer(buffer))
                buffer = [token]

    if buffer:
        merged_tokens.append(_combine_buffer(buffer))
    return merged_tokens

def _combine_buffer(buffer):
    first_token = buffer[0]
    last_token = buffer[-1]
    merged_word = "".join(t["word"].replace("##", "") for t in buffer)
    entity_group = first_token["entity_group"]
    merged_score = min(t["score"] for t in buffer)
    merged_start = first_token["start"]
    merged_end = last_token["end"]
    return {
        "word": merged_word,
        "entity_group": entity_group,
        "score": merged_score,
        "start": merged_start,
        "end": merged_end
    }

def format_ner_results(chopped_results):
    lines = []
    for item in chopped_results:
        entity = item.get("word", "")
        entity_group = item.get("entity_group", "")
        score = item.get("score", 0)
        lines.append(f"Entity: {entity}, Entity Group: {entity_group}, Score: {score:.4f}")
    return "\n".join(lines)

def sort_by_confidence(ner_results):
    return sorted(ner_results, key=lambda x: x.get("score", 0), reverse=True)

# ----------------------------------------------------------------------------------------------------------------------
# (4) Context Snippet Extraction (2 words before & after)
# ----------------------------------------------------------------------------------------------------------------------
def get_context_snippet(doc, entity_start, entity_end, window=2):
    """
    Using a spaCy Doc object (from the original text!), find the token indices overlapping
    with [entity_start, entity_end) and retrieve up to `window` tokens before and after.
    Return the joined text snippet.
    """
    start_token_idx = None
    end_token_idx = None

    for i, token in enumerate(doc):
        token_char_start = token.idx
        token_char_end = token.idx + len(token.text)

        # If the entity_start falls within this token
        if start_token_idx is None and (token_char_start <= entity_start < token_char_end):
            start_token_idx = i

        # If the entity_end falls within this token
        if end_token_idx is None and (token_char_start < entity_end <= token_char_end):
            end_token_idx = i
        
        if start_token_idx is not None and end_token_idx is not None:
            break

    # If for some reason we couldn't map it exactly, fallback to substring
    if start_token_idx is None or end_token_idx is None:
        return doc.text[entity_start:entity_end].strip()

    snippet_start_idx = max(0, start_token_idx - window)
    snippet_end_idx = min(len(doc) - 1, end_token_idx + window)
    snippet_tokens = doc[snippet_start_idx : snippet_end_idx + 1]

    snippet = " ".join(t.text for t in snippet_tokens)
    return snippet.strip()

# ----------------------------------------------------------------------------------------------------------------------
# (5) Main Workflow
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Create the biomedical NER pipeline
    biomedical_ner = create_biomedical_ner_pipeline()

    # 2) Load the ICD-10 codes
    file_path = r"C:\Users\cerul\Downloads\icd10cm-codes-2025.txt"
    icd_codes = load_icd_codes(file_path)

    # 3) Sample clinical text
    test_text = """
    A 55-year-old female presents to the emergency department with an acute episode of shortness of breath.
    During her evaluation, she reports a persistent cough that she describes as causing a primary cough headache.
    She also notes the onset of characteristics suggestive of fever. Additionally, the patient
    experiences occasional dizziness, and she has felt intermittent palpitations throughout the day.
    On lung auscultation, Wheezing is detected, prompting further investigation.

    She is undergoing diagnostic tests, including an electrocardiogram and chest imaging, to determine the cause of
    these symptoms and guide subsequent management.
    Displacement of unspecified cardiac electronic device, sequela.
    """



    # 4) Create a spaCy doc from the ORIGINAL text for negation detection & snippet extraction
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(test_text)

    # 5) Run the NER pipeline on the ORIGINAL text
    ner_results = biomedical_ner(test_text)

    # 6) Filter by confidence
    ner_results = filter_by_confidence(ner_results, threshold=0.80)

    # 7) Merge subword tokens
    ner_results = merge_subword_tokens(ner_results)

    # 8) Sort by confidence
    ner_results = sort_by_confidence(ner_results)

    # 9) Filter to only relevant ICD-type entities
    ner_results = filter_icd_entities(ner_results)

    # 10) Remove negated entities using the original text
    #     This ensures offsets match the doc we created above.
    ner_results = remove_negated_entities(ner_results, test_text, debug=True)

    # 11) Display final NER results (after negation removal)
    print("\n=== NER Entities (After Negation Removal) ===")
    print(format_ner_results(ner_results))

    # 12) For each non-negated entity, retrieve the snippet and do fuzzy matching
    icd_mappings = []
    for ent in ner_results:
        entity_start = ent["start"]
        entity_end = ent["end"]

        snippet = get_context_snippet(doc, entity_start, entity_end, window=2)

        code, description, score = get_best_icd_match(snippet, icd_codes)
        if code:
            icd_mappings.append({
                "entity": ent["word"],
                "context_snippet": snippet,
                "icd_code": code,
                "icd_description": description,
                "match_score": score
            })

    # 13) Print ICD-10 mappings with snippet info
    print("\n=== ICD-10 Mappings ===")
    for mapping in icd_mappings:
        print(f"Entity: {mapping['entity']}")
        print(f"  Context Snippet: {mapping['context_snippet']}")
        print(f"  ICD Code: {mapping['icd_code']}")
        print(f"  Description: {mapping['icd_description']}")
        print(f"  Match Score: {mapping['match_score']}")
        print("---")
