from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GUIDELINES_DIR = ROOT / "guidelines"
METADATA_CSV = ROOT / "metadata" / "guidelines.csv"
OUTPUT_CSV = ROOT / "metadata" / "guidelines_new_rows_template.csv"

# Read existing metadata if present
if METADATA_CSV.exists():
    existing = pd.read_csv(METADATA_CSV).fillna("")
else:
    existing = pd.DataFrame(columns=[
        "filename", "title", "specialty", "status",
        "version", "approval_date", "review_date", "owner"
    ])

existing_filenames = set(existing["filename"].astype(str).str.strip())

rows = []
for path in sorted(GUIDELINES_DIR.iterdir()):
    if not path.is_file():
        continue

    if path.name == "README.md":
        continue

    if path.name in existing_filenames:
        continue

    stem = path.stem.replace("-", " ").replace("_", " ")
    title = " ".join(word for word in stem.split())

    rows.append({
        "filename": path.name,
        "title": title,
        "specialty": "",
        "status": "Draft",
        "version": "",
        "approval_date": "",
        "review_date": "",
        "owner": ""
    })

new_rows = pd.DataFrame(rows)

if new_rows.empty:
    print("No new files found.")
else:
    new_rows.to_csv(OUTPUT_CSV, index=False)
    print(f"Created template: {OUTPUT_CSV}")
    print(f"Rows: {len(new_rows)}")