import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\- ']", " ", text)
    text = text.replace("--", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text):
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def detokenize(tokens):
    return " ".join(token for token in tokens if token).strip()


def levenshtein_distance(seq_a, seq_b):
    if seq_a == seq_b:
        return 0
    if len(seq_a) == 0:
        return len(seq_b)
    if len(seq_b) == 0:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        curr = [i]
        for j, item_b in enumerate(seq_b, start=1):
            substitution = prev[j - 1] + int(item_a != item_b)
            insertion = curr[j - 1] + 1
            deletion = prev[j] + 1
            curr.append(min(substitution, insertion, deletion))
        prev = curr
    return prev[-1]


class PhonemeConverter:
    def __init__(self):
        from g2p_en import G2p

        self._g2p = G2p()
        self._cache = {}

    def text_to_phonemes(self, text):
        normalized = normalize_text(text)
        if normalized in self._cache:
            return self._cache[normalized]

        phonemes = []
        for phone in self._g2p(normalized):
            if phone == " ":
                phonemes.append("SIL")
                continue
            phone = re.sub(r"[0-9]", "", str(phone))
            if re.fullmatch(r"[A-Z]+", phone):
                phonemes.append(phone)

        self._cache[normalized] = phonemes
        return phonemes


def sanitize_candidate_text(text):
    text = str(text).replace(">", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_diff_ranges(anchor_tokens, other_tokens):
    matcher = SequenceMatcher(a=anchor_tokens, b=other_tokens, autojunk=False)
    ranges = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        start = i1
        end = i2 if i2 > i1 else min(len(anchor_tokens), i1 + 1)
        ranges.append((start, end))
    return ranges


def merge_ranges(ranges):
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        prev = merged[-1]
        if start <= prev[1]:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def extract_phrase_for_span(anchor_tokens, candidate_tokens, span_start, span_end):
    matcher = SequenceMatcher(a=anchor_tokens, b=candidate_tokens, autojunk=False)
    collected = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            if span_start <= i1 <= span_end:
                collected.extend(candidate_tokens[j1:j2])
            continue

        overlap_start = max(span_start, i1)
        overlap_end = min(span_end, i2)
        if overlap_start >= overlap_end:
            continue

        if tag == "equal":
            offset = overlap_start - i1
            length = overlap_end - overlap_start
            collected.extend(candidate_tokens[j1 + offset : j1 + offset + length])
        elif tag == "replace":
            collected.extend(candidate_tokens[j1:j2])
        elif tag == "delete":
            continue
    return detokenize(collected)


def extract_confusion_spans(candidate_texts, phoneme_converter=None, top_k=5, context_window=3):
    candidate_texts = [sanitize_candidate_text(text) for text in candidate_texts[:top_k] if text]
    if len(candidate_texts) < 2:
        return []

    tokenized = [tokenize_text(text) for text in candidate_texts]
    anchor_tokens = tokenized[0]
    diff_ranges = []
    for candidate_tokens in tokenized[1:]:
        diff_ranges.extend(extract_diff_ranges(anchor_tokens, candidate_tokens))

    spans = []
    for span_start, span_end in merge_ranges(diff_ranges):
        alternatives = []
        seen_alternatives = {}
        for candidate_index, candidate_tokens in enumerate(tokenized):
            phrase = extract_phrase_for_span(anchor_tokens, candidate_tokens, span_start, span_end)
            if phrase not in seen_alternatives:
                phonemes = phoneme_converter.text_to_phonemes(phrase) if phoneme_converter else []
                seen_alternatives[phrase] = {
                    "text": phrase,
                    "phonemes": phonemes,
                    "candidate_indices": [],
                }
            seen_alternatives[phrase]["candidate_indices"].append(candidate_index)

        alternatives = list(seen_alternatives.values())
        if len(alternatives) <= 1:
            continue

        pairwise_distances = []
        for idx in range(len(alternatives)):
            for jdx in range(idx + 1, len(alternatives)):
                pairwise_distances.append(
                    levenshtein_distance(
                        alternatives[idx]["phonemes"],
                        alternatives[jdx]["phonemes"],
                    )
                )

        spans.append(
            {
                "start": span_start,
                "end": span_end,
                "anchor_text": detokenize(anchor_tokens[span_start:span_end]),
                "left_context": detokenize(anchor_tokens[max(0, span_start - context_window) : span_start]),
                "right_context": detokenize(anchor_tokens[span_end : span_end + context_window]),
                "alternatives": alternatives,
                "min_phoneme_distance": min(pairwise_distances) if pairwise_distances else 0,
                "max_phoneme_distance": max(pairwise_distances) if pairwise_distances else 0,
            }
        )
    return spans


def should_trigger_retrieval(confusion_spans, score_margin, phoneme_distance_threshold, score_margin_threshold):
    if not confusion_spans:
        return False
    if score_margin > score_margin_threshold:
        return False
    min_distance = min(span["min_phoneme_distance"] for span in confusion_spans)
    return min_distance <= phoneme_distance_threshold


def summarize_confusion_set(confusion_spans):
    summary = []
    for span in confusion_spans:
        summary.append(
            {
                "span": [span["start"], span["end"]],
                "anchor_text": span["anchor_text"],
                "alternatives": [alt["text"] for alt in span["alternatives"]],
                "min_phoneme_distance": span["min_phoneme_distance"],
            }
        )
    return summary


def build_analysis_record(
    utterance_index,
    partition,
    reference,
    ranked_candidates,
    raw_nbest,
    phoneme_converter,
    confusion_top_k=5,
    include_confusion=True,
):
    candidate_texts = [candidate["text"] for candidate in ranked_candidates]
    confusion_spans = []
    if include_confusion:
        confusion_spans = extract_confusion_spans(
            candidate_texts,
            phoneme_converter=phoneme_converter,
            top_k=confusion_top_k,
        )
    top_score = ranked_candidates[0]["total_score"] if ranked_candidates else float("-inf")
    second_score = ranked_candidates[1]["total_score"] if len(ranked_candidates) > 1 else float("-inf")
    score_margin = float(top_score - second_score) if len(ranked_candidates) > 1 else float("inf")
    return {
        "utterance_index": int(utterance_index),
        "partition": partition,
        "reference": reference,
        "ranked_candidates": ranked_candidates,
        "raw_nbest": [
            {
                "text": sanitize_candidate_text(entry[0]),
                "acoustic_score": float(entry[1]),
                "old_lm_score": float(entry[2]),
            }
            for entry in raw_nbest
        ],
        "confusion_spans": confusion_spans,
        "score_margin": score_margin,
    }


def write_jsonl(path, records):
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def read_jsonl(path):
    records = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass
class BM25Document:
    doc_id: str
    text: str
    tokens: list
    metadata: dict


class SimpleBM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_freq = Counter()
        self.doc_len = []
        self.avg_doc_len = 0.0

        for doc in documents:
            unique_tokens = set(doc.tokens)
            for token in unique_tokens:
                self.doc_freq[token] += 1
            self.doc_len.append(len(doc.tokens))

        self.avg_doc_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

    def _idf(self, token):
        n_docs = len(self.documents)
        freq = self.doc_freq.get(token, 0)
        return math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query, top_k=5):
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        results = []
        query_counts = Counter(query_tokens)
        for doc, doc_len in zip(self.documents, self.doc_len):
            term_counts = Counter(doc.tokens)
            score = 0.0
            for token, q_count in query_counts.items():
                if token not in term_counts:
                    continue
                tf = term_counts[token]
                idf = self._idf(token)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1.0))
                score += q_count * idf * (tf * (self.k1 + 1)) / max(denom, 1e-8)
            if score > 0:
                results.append(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "score": float(score),
                        "metadata": doc.metadata,
                    }
                )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


def load_corpus_documents(corpus_path):
    corpus_path = Path(corpus_path)
    docs = []
    for record in read_jsonl(corpus_path):
        text = record["text"]
        docs.append(
            BM25Document(
                doc_id=str(record["doc_id"]),
                text=text,
                tokens=tokenize_text(text),
                metadata={key: value for key, value in record.items() if key not in {"doc_id", "text"}},
            )
        )
    return docs


def build_retriever(corpus_path):
    return SimpleBM25Retriever(load_corpus_documents(corpus_path))


def build_retrieval_query(confusion_spans):
    query_terms = []
    for span in confusion_spans:
        if span["left_context"]:
            query_terms.append(span["left_context"])
        for alternative in span["alternatives"]:
            if alternative["text"]:
                query_terms.append(alternative["text"])
        if span["right_context"]:
            query_terms.append(span["right_context"])
    return " ".join(query_terms)


def retrieve_confusion_context(confusion_spans, retriever, top_k=5):
    if retriever is None:
        return []
    query = build_retrieval_query(confusion_spans)
    return retriever.search(query, top_k=top_k)


def replace_span(tokens, start, end, replacement_text):
    replacement_tokens = tokenize_text(replacement_text)
    return tokens[:start] + replacement_tokens + tokens[end:]


def expand_candidates(anchor_text, confusion_spans, retrieved_docs, max_new_candidates=3):
    if not confusion_spans or not retrieved_docs:
        return []

    anchor_tokens = tokenize_text(anchor_text)
    generated = []
    seen = {detokenize(anchor_tokens)}
    for span in confusion_spans:
        for doc in retrieved_docs:
            target_phrase = doc.get("metadata", {}).get("target_phrase", "")
            if not target_phrase:
                continue
            candidate_tokens = replace_span(anchor_tokens, span["start"], span["end"], target_phrase)
            candidate_text = detokenize(candidate_tokens)
            if candidate_text and candidate_text not in seen:
                seen.add(candidate_text)
                generated.append(candidate_text)
            if len(generated) >= max_new_candidates:
                return generated
    return generated


def aggregate_confusion_counts(records, top_k_candidates=5):
    counts = Counter()
    for record in records:
        spans = record.get("confusion_spans", [])
        for span in spans:
            alternatives = sorted(
                alternative["text"]
                for alternative in span.get("alternatives", [])
                if alternative.get("text")
            )
            if len(alternatives) >= 2:
                counts[tuple(alternatives)] += 1
    return counts


def collect_transcript_contexts(transcripts, phrase, max_docs=50):
    phrase_tokens = tokenize_text(phrase)
    if not phrase_tokens:
        return []

    contexts = []
    for transcript in transcripts:
        tokens = tokenize_text(transcript)
        phrase_len = len(phrase_tokens)
        for idx in range(0, len(tokens) - phrase_len + 1):
            if tokens[idx : idx + phrase_len] == phrase_tokens:
                left = detokenize(tokens[max(0, idx - 4) : idx])
                right = detokenize(tokens[idx + phrase_len : idx + phrase_len + 4])
                snippet = detokenize(tokens[max(0, idx - 6) : idx + phrase_len + 6])
                contexts.append(
                    {
                        "left_context": left,
                        "right_context": right,
                        "snippet": snippet,
                    }
                )
                if len(contexts) >= max_docs:
                    return contexts
    return contexts


def build_confusion_corpus(records, transcripts, phoneme_converter, top_k_confusions=100, docs_per_phrase=30):
    counts = aggregate_confusion_counts(records)
    corpus_docs = []
    ranked_confusions = counts.most_common(top_k_confusions)
    for confusion_index, (alternatives, frequency) in enumerate(ranked_confusions):
        confusion_key = " | ".join(alternatives)
        for phrase in alternatives:
            contexts = collect_transcript_contexts(transcripts, phrase, max_docs=docs_per_phrase)
            if not contexts:
                contexts = [{"left_context": "", "right_context": "", "snippet": phrase}]
            phonemes = phoneme_converter.text_to_phonemes(phrase)
            for context_idx, context in enumerate(contexts):
                corpus_docs.append(
                    {
                        "doc_id": f"{confusion_index}_{context_idx}_{normalize_text(phrase).replace(' ', '_')}",
                        "confusion_key": confusion_key,
                        "target_phrase": phrase,
                        "frequency": frequency,
                        "text": context["snippet"],
                        "source": "train_context",
                        "phonemes": phonemes,
                        "left_context": context["left_context"],
                        "right_context": context["right_context"],
                    }
                )

            context_terms = [ctx["left_context"] for ctx in contexts if ctx["left_context"]]
            context_terms.extend(ctx["right_context"] for ctx in contexts if ctx["right_context"])
            if context_terms:
                summary_text = (
                    f"target phrase {phrase} phonemes {' '.join(phonemes)} "
                    f"context examples {' ; '.join(context_terms[:5])}"
                )
                corpus_docs.append(
                    {
                        "doc_id": f"{confusion_index}_phoneme_{normalize_text(phrase).replace(' ', '_')}",
                        "confusion_key": confusion_key,
                        "target_phrase": phrase,
                        "frequency": frequency,
                        "text": summary_text,
                        "source": "phoneme_context",
                        "phonemes": phonemes,
                        "left_context": "",
                        "right_context": "",
                    }
                )
    return corpus_docs, ranked_confusions
