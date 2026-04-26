from __future__ import annotations

import re

from .schemas import TifaQuestion

COLORS = [
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
]

COUNT_WORDS = {
    "one": "one",
    "two": "two",
    "three": "three",
    "four": "four",
    "five": "five",
    "six": "six",
    "seven": "seven",
    "eight": "eight",
    "nine": "nine",
    "ten": "ten",
    "several": "several",
    "many": "many",
    "few": "a few",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "near",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

PEOPLE_WORDS = {
    "boy",
    "child",
    "children",
    "face",
    "girl",
    "hand",
    "hands",
    "human",
    "man",
    "men",
    "person",
    "people",
    "portrait",
    "woman",
    "women",
}

ANIMAL_WORDS = {
    "bird",
    "cat",
    "cow",
    "dog",
    "elephant",
    "giraffe",
    "horse",
    "sheep",
    "zebra",
}

STYLE_WORDS = {
    "anime",
    "cartoon",
    "digital",
    "drawing",
    "illustration",
    "oil",
    "painting",
    "render",
    "sketch",
    "watercolor",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]*|\d+", text)]


def _clean_phrase(phrase: str) -> str:
    words = [
        word
        for word in _tokens(phrase)
        if word not in STOPWORDS and len(word) > 1
    ]
    return " ".join(words[:4])


def _choice_pool(correct: str, pool: list[str], n: int = 4) -> list[str]:
    choices = [correct]
    for value in pool:
        if value != correct and value not in choices:
            choices.append(value)
        if len(choices) >= n:
            break
    return choices


def _add_unique(out: list[TifaQuestion], question: TifaQuestion):
    key = (question.question.lower(), question.answer.lower())
    existing = {(q.question.lower(), q.answer.lower()) for q in out}
    if key not in existing:
        out.append(question)


def _content_terms(tokens: list[str]) -> list[str]:
    out = []
    seen = set()
    for token in tokens:
        if token in STOPWORDS or token in COLORS or token in COUNT_WORDS:
            continue
        if len(token) < 3 or token.isdigit() or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def generate_heuristic_questions(caption: str, max_questions: int = 12) -> list[TifaQuestion]:
    """Generate a small offline TIFA-like question set from surface prompt cues.

    This is intentionally conservative: it produces mostly yes/no existence,
    color, count, relation, and common Stable Diffusion artifact checks. It is
    weaker than LLM-generated TIFA questions, but good enough for offline smoke
    tests and quota-free exploratory runs.
    """
    tokens = _tokens(caption)
    token_set = set(tokens)
    questions: list[TifaQuestion] = []

    for term in _content_terms(tokens)[:6]:
        element_type = "animal/human" if term in PEOPLE_WORDS or term in ANIMAL_WORDS else "object"
        _add_unique(
            questions,
            TifaQuestion(
                element=term,
                element_type=element_type,
                question=f"Is there a {term} in the image?",
                choices=["yes", "no"],
                answer="yes",
            ),
        )

    for match in re.finditer(
        r"\b(" + "|".join(COLORS) + r")\b\s+(?:\w+\s+){0,2}?([A-Za-z][A-Za-z'-]*)",
        caption.lower(),
    ):
        color = match.group(1)
        obj = _clean_phrase(match.group(2))
        if not obj:
            continue
        _add_unique(
            questions,
            TifaQuestion(
                element=obj,
                element_type="color",
                question=f"What color is the {obj}?",
                choices=_choice_pool(color, COLORS),
                answer=color,
            ),
        )

    for i, token in enumerate(tokens[:-1]):
        count = COUNT_WORDS.get(token, token if token.isdigit() else "")
        if not count:
            continue
        obj = tokens[i + 1]
        if obj in STOPWORDS:
            continue
        _add_unique(
            questions,
            TifaQuestion(
                element=obj,
                element_type="counting",
                question=f"Are there {count} {obj} in the image?",
                choices=["yes", "no"],
                answer="yes",
            ),
        )

    relation_patterns = [
        (r"\b(.+?)\s+on\s+(.+?)(?:[,.]|$)", "spatial", "Is {left} on {right}?"),
        (r"\b(.+?)\s+near\s+(.+?)(?:[,.]|$)", "spatial", "Is {left} near {right}?"),
        (r"\b(.+?)\s+with\s+(.+?)(?:[,.]|$)", "attribute", "Does the image show {left} with {right}?"),
    ]
    for pattern, element_type, template in relation_patterns:
        for match in re.finditer(pattern, caption.lower()):
            left = _clean_phrase(match.group(1).split()[-3:])
            right = _clean_phrase(match.group(2).split()[:3])
            if left and right:
                _add_unique(
                    questions,
                    TifaQuestion(
                        element=f"{left} {right}",
                        element_type=element_type,
                        question=template.format(left=left, right=right),
                        choices=["yes", "no"],
                        answer="yes",
                    ),
                )

    if token_set & PEOPLE_WORDS:
        _add_unique(
            questions,
            TifaQuestion(
                element="fingers",
                element_type="sd_artifact",
                question="Does the person have the normal number of fingers?",
                choices=["yes", "no"],
                answer="yes",
            ),
        )
        _add_unique(
            questions,
            TifaQuestion(
                element="anatomy",
                element_type="sd_artifact",
                question="Are the limbs and body proportions anatomically normal?",
                choices=["yes", "no"],
                answer="yes",
            ),
        )

    if not (token_set & STYLE_WORDS):
        _add_unique(
            questions,
            TifaQuestion(
                element="realism",
                element_type="sd_artifact",
                question="Does the image look photorealistic?",
                choices=["yes", "no"],
                answer="yes",
            ),
        )

    if token_set & {"label", "license", "plate", "sign", "text"}:
        _add_unique(
            questions,
            TifaQuestion(
                element="text",
                element_type="sd_artifact",
                question="Is the text in the image legible and correctly spelled?",
                choices=["yes", "no"],
                answer="yes",
            ),
        )

    return questions[:max_questions]
