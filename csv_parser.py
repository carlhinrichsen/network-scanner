import pandas as pd
import io
from typing import Union

def parse_linkedin_csv(content: Union[bytes, str]) -> list[dict]:
    """
    Parse LinkedIn connections CSV export.
    Handles the 3-row header (notes + blank + actual headers).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    # LinkedIn exports have 3 preamble lines before actual headers
    lines = content.splitlines()

    # Find the actual header row (contains "First Name")
    header_idx = 0
    for i, line in enumerate(lines):
        if "First Name" in line:
            header_idx = i
            break

    clean_csv = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(clean_csv))

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    column_map = {
        "First Name": "first_name",
        "Last Name": "last_name",
        "URL": "linkedin_url",
        "Email Address": "email",
        "Company": "company",
        "Position": "position",
        "Connected On": "connected_on",
    }

    df = df.rename(columns=column_map)

    # Keep only known columns
    keep = [v for v in column_map.values() if v in df.columns]
    df = df[keep]

    # Clean up
    df = df.fillna("")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop rows without a LinkedIn URL
    df = df[df["linkedin_url"].str.startswith("http")]

    return df.to_dict(orient="records")
