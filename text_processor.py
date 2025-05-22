import spacy
from fastcoref import FCoref
from typing import List
import re, os
from googletrans import Translator
from dotenv import load_dotenv
import requests
from fastapi import HTTPException

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")


nlp = spacy.load("en_core_web_lg")
coref_model = FCoref()
CAPITALIZED_PRONOUNS = {
    "He",
    "She",
    "His",
    "Her",
    "They",
    "Their",
    "It",
    "Its",
    "You",
    "Your",
    "I",
    "We",
    "Our",
}


def clean_caption(line: str) -> str:
    # Remove leading numbers, dashes, bullets, bold text
    line = re.sub(r"^\s*\d+[\.\-)]?\s*", "", line)  # "1. " or "2-" or "3)"
    line = re.sub(r"^\*\*(.*?)\*\*\s*[-–—]?\s*", "", line)  # "**Title** —"
    return line.strip()


def get_script_captions(script_text: str):
    API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You convert story or movie scripts into detailed, visually rich image generation captions. "
                "Each caption should describe a visually distinct scene as if it were to be illustrated or rendered, "
                "but do not number them or include titles — just full, descriptive sentences."
                "add the appropriate camera shot or angle each sentence."
                "these are the 8 shots you can use, close up shot, extreme close up shot, long shot, low angle shot, high angle shot, dutch angle, over the shoulder shot, medium shot."
            ),
        },
        {
            "role": "user",
            "content": f"SCRIPT:\n{script_text}\n\nReturn only image generation captions, one per line, no numbering, no headings.",
        },
    ]

    payload = {
        "model": "deepseek/deepseek-v3-0324",
        "messages": messages,
        "temperature": 0.7,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        json_data = response.json()
        raw_output = json_data["choices"][0]["message"]["content"]
        lines = raw_output.strip().split("\n")
        return [clean_caption(line) for line in lines if line.strip()]
    else:
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek API error: {response.status_code} - {response.text}",
        )


def is_capitalized_pronoun(span: spacy.tokens.Span, text: str) -> bool:
    """Check if the span is a single capitalized pronoun in the original text."""
    if len(span) != 1 or span[0].pos_ != "PRON":
        return False
    # Use original casing from text
    start = span.start_char
    end = span.end_char
    original_token_text = text[start:end]
    return original_token_text[0].isupper()


def get_fastcoref_clusters(doc, text):
    preds = coref_model.predict(texts=[text])
    fast_clusters = preds[0].get_clusters(as_strings=False)

    converted_clusters = []
    for cluster in fast_clusters:
        new_cluster = []
        for start_char, end_char in cluster:
            span = doc.char_span(start_char, end_char)
            if span is not None:
                new_cluster.append((span.start, span.end))
        if new_cluster:
            converted_clusters.append(new_cluster)

    return converted_clusters


def get_span_noun_indices(doc: spacy.tokens.Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0] : span[1]] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    return [
        i
        for i, span_pos in enumerate(spans_pos)
        if any(pos in ["NOUN", "PROPN"] for pos in span_pos)
    ]


def get_cluster_head(
    doc: spacy.tokens.Doc, cluster: List[List[int]], noun_indices: List[int]
):
    head_idx = noun_indices[0] if noun_indices else 0
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end]
    return head_span, (head_start, head_end)


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any(s != span and s[0] >= span[0] and s[1] <= span[1] for s in all_spans)


def replace_coref_span(doc, coref_span, resolved_text, mention_span):
    start, end = coref_span
    prefix = " " if start > 0 and not doc[start - 1].whitespace_ else ""
    suffix = doc[end - 1].whitespace_ if end < len(doc) else ""

    resolved_text[start] = prefix + mention_span.text + suffix
    for i in range(start + 1, end):
        resolved_text[i] = ""


def improved_replace_corefs(
    doc: spacy.tokens.Doc, clusters: List[List[List[int]]], text: str
):
    resolved = [token.text_with_ws for token in doc]
    all_spans = [span for cluster in clusters for span in cluster]

    for cluster in clusters:
        noun_indices = get_span_noun_indices(doc, cluster)
        if not noun_indices:
            continue

        mention_span, mention = get_cluster_head(doc, cluster, noun_indices)

        for coref in cluster:
            coref_span = doc[coref[0] : coref[1]]
            if (
                coref != mention
                and not is_containing_other_spans(coref, all_spans)
                and is_capitalized_pronoun(coref_span, text)
            ):
                replace_coref_span(doc, coref, resolved, mention_span)

    return "".join(resolved)


def detect_and_translate_to_english(text: str) -> str:
    try:
        translator = Translator()
        detected = translator.detect(text)
        if detected.lang == "tl":
            print("[Info] Detected language: Filipino (tl). Translating to English...")
            translated = translator.translate(text, src="tl", dest="en")
            return translated.text
        return text
    except Exception as e:
        print(f"[Warning] Language detection or translation failed: {e}")
        return text


def resolve_coreferences(text: str) -> str:
    doc = nlp(text)
    clusters = get_fastcoref_clusters(doc, text)
    return improved_replace_corefs(doc, clusters, text)


def remove_dialogues(text: str) -> str:
    text = re.sub(r'(["“\']).*?\1', "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def get_resolved_sentences(text: str) -> List[str]:
    text = detect_and_translate_to_english(text)
    resolved_text = resolve_coreferences(text)
    no_dialogue_text = remove_dialogues(resolved_text)
    resolved_doc = nlp(no_dialogue_text)
    return [sent.text.strip() for sent in resolved_doc.sents]
