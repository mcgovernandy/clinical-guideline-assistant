from pathlib import Path
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

ROOT = Path(__file__).resolve().parents[1]
GUIDELINES_DIR = ROOT / "guidelines"
METADATA_CSV = ROOT / "metadata" / "guidelines.csv"
OUTPUT_JSON = ROOT / "ingestion" / "vector_store_info.json"

ALLOWED_LEVELS = ["local", "national", "international"]


def normalise_level(value: str) -> str:
    return str(value).strip().lower()


def create_vector_store(name: str):
    return client.vector_stores.create(name=name)


def upload_file_to_store(vector_store_id: str, file_path: Path):
    with open(file_path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="assistants")
    client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store_id,
        file_id=uploaded.id,
    )
    return uploaded.id


def main():
    df = pd.read_csv(METADATA_CSV).fillna("")
    df = df[df["status"].astype(str).str.strip().str.lower() == "approved"].copy()

    if df.empty:
        raise ValueError("No approved guidelines found in metadata/guidelines.csv")

    if "guideline_level" not in df.columns:
        raise ValueError("metadata/guidelines.csv must contain a guideline_level column")

    df["guideline_level"] = df["guideline_level"].map(normalise_level)

    invalid_levels = sorted(set(df[~df["guideline_level"].isin(ALLOWED_LEVELS)]["guideline_level"].tolist()))
    if invalid_levels:
        raise ValueError(f"Invalid guideline_level values found: {invalid_levels}")

    stores = {}
    summary = {
        "app_name": "Sally",
        "assistant_codename": "SAL 9000",
        "vector_stores": {},
        "uploaded_files": [],
    }

    for level in ALLOWED_LEVELS:
        level_df = df[df["guideline_level"] == level].copy()
        if level_df.empty:
            print(f"No approved {level} guidelines found. Skipping store creation.")
            continue

        store = create_vector_store(name=f"Sally Guidelines - {level.title()}")
        stores[level] = store.id
        summary["vector_stores"][level] = {
            "vector_store_id": store.id,
            "name": store.name,
            "file_count": 0,
            "files": [],
        }

        print(f"\nCreated {level} vector store: {store.id}")

        for _, row in level_df.iterrows():
            file_path = GUIDELINES_DIR / str(row["filename"]).strip()
            if not file_path.exists():
                print(f"WARNING: file not found, skipping: {file_path}")
                continue

            print(f"Uploading [{level}] {file_path.name}")
            openai_file_id = upload_file_to_store(store.id, file_path)

            record = {
                "filename": str(row["filename"]).strip(),
                "title": str(row.get("title", "")).strip(),
                "guideline_level": level,
                "approval_date": str(row.get("approval_date", "")).strip(),
                "review_date": str(row.get("review_date", "")).strip(),
                "owner": str(row.get("owner", "")).strip(),
                "openai_file_id": openai_file_id,
            }
            summary["vector_stores"][level]["files"].append(record)
            summary["uploaded_files"].append(record)
            summary["vector_stores"][level]["file_count"] += 1

    if not summary["vector_stores"]:
        raise ValueError("No vector stores were created. Check your metadata and guidelines folder.")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved vector store info to: {OUTPUT_JSON}")
    print(json.dumps(summary["vector_stores"], indent=2))


if __name__ == "__main__":
    main()
