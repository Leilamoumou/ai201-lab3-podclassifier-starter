import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """



    lines = [
            "You are classifying podcast episodes by their format.",
            f"Classify the episode into exactly one of: {', '.join(VALID_LABELS)}.",
            "",
            "- interview: a host in conversation/Q&A with one or more guests",
            "- solo: one host speaking from their own memory, experience, or opinion; no guests, nothing assembled from outside",
            "- panel: multiple guests with roughly equal speaking time discussing or debating",
            "- narrative: a story assembled from external sources (interviews, archival audio, reporting) into an arc, even if one voice narrates",
            "",
            "Here are labeled examples:",
            "",
        ]
    for ex in labeled_examples:
            lines.append(f"Title: {ex['title']}")
            lines.append(f"Description: {ex['description']}")
            lines.append(f"Label: {ex['label']}")
            lines.append("---")

    lines += [
            "",
            "Now classify this episode:",
            "",
            f"Description: {description}",
            "",
            "Respond with the label on the FIRST line by itself (one of: "
            f"{', '.join(VALID_LABELS)}), then one sentence of reasoning on the next line.",
        ]
    return "\n".join(lines)



def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """

    prompt = build_few_shot_prompt(labeled_examples, description)
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        text = response.choices[0].message.content
        parts = text.strip().splitlines()
        first = "".join(c for c in parts[0].lower() if c.isalpha())  # "**Interview**" -> "interview"
        if first in VALID_LABELS:
            label = first
        else:
            label = next((l for l in VALID_LABELS if l in text.lower()), "unknown")
        
        reasoning = " ".join(parts[1:]).strip() or text.strip()
        return{"label": label, "reasoning": reasoning}
    
    except Exception as e:
        return {"label": "unknown", "reasoning": f"error: {e}"}
