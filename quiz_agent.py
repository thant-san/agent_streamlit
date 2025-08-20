from __future__ import annotations

import random
import re
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nltk
from PyPDF2 import PdfReader


@dataclass
class SMTPConfig:
    """Holds SMTP configuration for sending emails."""

    host: str
    port: int
    username: str
    password: str
    use_tls: bool = True


def _ensure_nltk_models() -> None:
    """Ensure required NLTK models are available at runtime.

    This downloads models only if missing to avoid repeated network calls.
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Newer NLTK uses 'punkt_tab' alongside 'punkt'
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            # Some versions may not provide punkt_tab; safe to ignore
            pass

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")


def _read_pdf_text(pdf_path: str | Path) -> str:
    """Extract text content from a PDF file."""

    reader = PdfReader(str(pdf_path))
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
    raw_text = "\n".join(pages_text)

    # Basic cleanup
    cleaned = re.sub(r"\s+", " ", raw_text)
    return cleaned.strip()


def _select_candidate_sentences(text: str) -> List[str]:
    """Split text to sentences and keep reasonably informative ones."""

    _ensure_nltk_models()
    sent_tokenize = getattr(nltk, "sent_tokenize", None)
    if sent_tokenize is None:
        from nltk.tokenize import sent_tokenize  # type: ignore

    sentences = nltk.sent_tokenize(text)
    filtered: List[str] = []
    for sentence in sentences:
        words = re.findall(r"\w+", sentence)
        if 8 <= len(words) <= 35 and not sentence.strip().endswith(":"):
            filtered.append(sentence.strip())
    return filtered


def _collect_nouns(text: str) -> List[str]:
    """Return a list of unique nouns from text (basic POS tagging)."""

    _ensure_nltk_models()
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    nouns = [w for w, t in tagged if t.startswith("NN") and re.match(r"^[A-Za-z][A-Za-z\-]+$", w)]
    # Normalize and deduplicate while preserving order
    seen = set()
    unique_nouns: List[str] = []
    for noun in nouns:
        base = noun.strip().strip(".,;:!?()[]{}'\"")
        if not base:
            continue
        key = base.lower()
        if key not in seen and len(base) > 2:
            seen.add(key)
            unique_nouns.append(base)
    return unique_nouns


def _mask_target_in_sentence(sentence: str, target: str) -> str:
    """Replace the first case-insensitive occurrence of target with a blank."""

    pattern = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE)
    return pattern.sub("_____", sentence, count=1)


def _build_mcq(
    sentence: str, target: str, noun_pool: Sequence[str], rng: random.Random
) -> Tuple[str, List[str], str]:
    """Build one multiple-choice question.

    Returns (question_text, options, correct_option)
    """

    # Distractors: choose 3 distinct nouns not equal to the target
    distractors_pool = [n for n in noun_pool if n.lower() != target.lower() and n.lower() != target.lower() + "s"]
    distractors = rng.sample(distractors_pool, k=min(3, len(distractors_pool)))
    all_options = distractors + [target]
    rng.shuffle(all_options)
    question_text = _mask_target_in_sentence(sentence, target)
    return question_text, all_options, target


def _format_quiz(qa_items: List[Tuple[str, List[str], str]]) -> str:
    """Format quiz as a plain-text email-friendly body."""

    lines: List[str] = []
    lines.append("Quiz")
    lines.append("")
    for idx, (question, options, correct) in enumerate(qa_items, start=1):
        lines.append(f"Q{idx}. {question}")
        option_labels = ["A", "B", "C", "D", "E", "F"]
        for opt_idx, option in enumerate(options):
            label = option_labels[opt_idx] if opt_idx < len(option_labels) else f"{opt_idx+1}"
            lines.append(f"   {label}) {option}")
        lines.append("")
    lines.append("Answer Key")
    for idx, (question, options, correct) in enumerate(qa_items, start=1):
        try:
            correct_index = options.index(correct)
        except ValueError:
            correct_index = 0
        label = ["A", "B", "C", "D", "E", "F"][correct_index]
        lines.append(f"Q{idx}: {label}")
    return "\n".join(lines)


def generate_quiz_from_pdf(
    pdf_path: str | Path,
    num_questions: int = 5,
    seed: Optional[int] = None,
) -> str:
    """Generate a multiple-choice quiz from the given PDF.

    - Extracts text from the PDF
    - Selects informative sentences
    - Masks a noun in each selected sentence to form a blank
    - Builds MCQs with distractors sampled from other nouns in the document

    Returns a formatted quiz body suitable for email.
    """

    rng = random.Random(seed)
    text = _read_pdf_text(pdf_path)
    if not text:
        raise ValueError("No text could be extracted from the PDF.")

    candidate_sentences = _select_candidate_sentences(text)
    if not candidate_sentences:
        raise ValueError("Could not find suitable sentences in the PDF to generate questions.")

    nouns = _collect_nouns(text)
    if not nouns:
        # Fallback: pick mid-length words as pseudo-nouns
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)]
        nouns = list(dict.fromkeys(words))

    # Prefer sentences that actually contain nouns
    chosen: List[Tuple[str, List[str], str]] = []
    seen_sentence_indexes = set()
    attempts = 0
    max_attempts = max(num_questions * 6, 30)
    while len(chosen) < num_questions and attempts < max_attempts:
        attempts += 1
        if not candidate_sentences:
            break
        idx = rng.randrange(0, len(candidate_sentences))
        if idx in seen_sentence_indexes:
            continue
        seen_sentence_indexes.add(idx)
        sentence = candidate_sentences[idx]

        # Identify nouns present in the sentence
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_tags = nltk.pos_tag(sentence_tokens)
        sentence_nouns = [w for w, t in sentence_tags if t.startswith("NN")]
        # Filter nouns to those appearing in the global pool
        sentence_nouns = [n for n in sentence_nouns if any(n.lower() == g.lower() for g in nouns)]

        if not sentence_nouns:
            continue

        target = rng.choice(sentence_nouns)
        question_text, options, correct = _build_mcq(sentence, target, nouns, rng)

        # Ensure options are unique and reasonably distinct
        normalized = []
        seen = set()
        for opt in options:
            key = opt.strip().lower()
            if key and key not in seen:
                seen.add(key)
                normalized.append(opt)
        if len(normalized) < 2:
            continue
        # Ensure correct is present
        if correct not in normalized:
            normalized.append(correct)
        # Limit to max 4 options
        options_final = normalized[:4]
        if correct not in options_final:
            # Replace last with correct if needed
            if options_final:
                options_final[-1] = correct
            else:
                options_final = [correct]
        rng.shuffle(options_final)

        chosen.append((question_text, options_final, correct))

    if not chosen:
        raise ValueError("Failed to generate quiz questions from the provided PDF.")

    return _format_quiz(chosen)


def agent_mode_send_quiz(
    pdf_path: str | Path,
    to_email: str,
    subject: str,
    smtp_config: SMTPConfig,
    num_questions: int = 5,
    seed: Optional[int] = None,
) -> None:
    """Generate a quiz from `pdf_path` and email it to `to_email` with `subject`.

    The email body is the generated quiz. Uses the given SMTP settings.
    """

    quiz_body = generate_quiz_from_pdf(pdf_path=pdf_path, num_questions=num_questions, seed=seed)

    message = EmailMessage()
    message["From"] = smtp_config.username
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(quiz_body)

    if smtp_config.use_tls:
        with smtplib.SMTP(smtp_config.host, smtp_config.port) as server:
            server.starttls()
            server.login(smtp_config.username, smtp_config.password)
            server.send_message(message)
    else:
        with smtplib.SMTP_SSL(smtp_config.host, smtp_config.port) as server:
            server.login(smtp_config.username, smtp_config.password)
            server.send_message(message)


__all__ = [
    "SMTPConfig",
    "generate_quiz_from_pdf",
    "agent_mode_send_quiz",
]

