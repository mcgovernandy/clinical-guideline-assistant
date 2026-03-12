import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

ROOT = Path(__file__).resolve().parents[1]
VECTOR_INFO = ROOT / "ingestion" / "vector_store_info.json"

with open(VECTOR_INFO, encoding="utf-8") as f:
    vector_info = json.load(f)

vector_store_id = vector_info["vector_store_id"]

GUIDELINE_RESPONSE_SCHEMA = {
    "name": "guideline_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean"
            },
            "answer": {
                "type": "string"
            },
            "quotes": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "source": {
                "type": "string"
            },
            "section_heading": {
                "type": "string"
            },
            "follow_up_needed": {
                "type": "boolean"
            },
            "related_guidelines": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "conflict_check": {
                "type": "string"
            },
            "notes": {
                "type": "string"
            }
        },
        "required": [
            "supported",
            "answer",
            "quotes",
            "source",
            "section_heading",
            "follow_up_needed",
            "related_guidelines",
            "conflict_check",
            "notes"
        ],
        "additionalProperties": False
    }
}


def ask(question: str):
    try:
        instructions = """
You are a hospital clinical guideline assistant.

Use only retrieved guideline content from file search.
Do not use background medical knowledge.
If the answer is not clearly supported by the guideline set, set supported=false.

Rules:
1. answer must be short, clear, and directly grounded in the retrieved guideline text.
2. quotes must contain verbatim text copied from the guideline.
3. source must contain the main file name or guideline title used for the answer.
4. section_heading should contain the most relevant section title if identifiable from the retrieved content. If not identifiable, say "Not clearly identified".
5. follow_up_needed should be true if the retrieved content is partial, high-level, or does not provide enough operational detail for safe clinical use.
6. related_guidelines should list other retrieved or obviously relevant guideline files if they appear relevant to the topic. Otherwise return an empty list.
7. conflict_check should briefly state one of:
   - "No conflict identified"
   - "Potential overlap - check related guidelines"
   - "Potential conflict - review quoted sections"
8. notes should explain uncertainty, limitations, overlap, or missing operational detail.
9. If the answer is unsupported, answer exactly:
   "I could not find a supported answer in the current guideline set."
10. If unsupported, quotes should be an empty list.
11. If the guideline mentions an issue but does not provide the full operational detail, supported may still be true, but follow_up_needed should be true.
12. Be conservative. If there is pregnancy/peri-operative/inpatient overlap, mention it in related_guidelines or notes.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=question,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id]
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": GUIDELINE_RESPONSE_SCHEMA["name"],
                    "strict": GUIDELINE_RESPONSE_SCHEMA["strict"],
                    "schema": GUIDELINE_RESPONSE_SCHEMA["schema"]
                }
            }
        )

        result = json.loads(response.output_text)

        print("\nSupported:")
        print(result["supported"])

        print("\nAnswer:")
        print(result["answer"])

        print("\nQuoted guideline text:")
        if result["quotes"]:
            for quote in result["quotes"]:
                print(f'- "{quote}"')
        else:
            print("- None")

        print("\nSource:")
        print(result["source"])

        print("\nSection heading:")
        print(result["section_heading"])

        print("\nFollow-up needed:")
        print(result["follow_up_needed"])

        print("\nRelated guidelines:")
        if result["related_guidelines"]:
            for item in result["related_guidelines"]:
                print(f"- {item}")
        else:
            print("- None")

        print("\nConflict check:")
        print(result["conflict_check"])

        print("\nNotes:")
        print(result["notes"])

    except Exception as e:
        print("\nAPI call failed.")
        print(str(e))


if __name__ == "__main__":
    q = input("Ask a guideline question: ").strip()
    if not q:
        print("No question entered.")
    else:
        ask(q)