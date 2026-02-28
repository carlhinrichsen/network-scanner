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

    if header_idx == 0 and "First Name" not in lines[0]:
        raise ValueError(
            "This doesn't look like a LinkedIn connections export. "
            "Please go to LinkedIn → Settings → Data Privacy → Get a copy of your data "
            "→ Connections, then upload the downloaded CSV."
        )

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

    # Validate that the essential URL column is present
    if "linkedin_url" not in df.columns:
        raise ValueError(
            "The CSV is missing the 'URL' column. "
            "Please make sure you're uploading a LinkedIn connections export "
            "(not invitations, messages, or another LinkedIn data file)."
        )

    # Keep only known columns
    keep = [v for v in column_map.values() if v in df.columns]
    df = df[keep]

    # Clean up
    df = df.fillna("")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop rows without a LinkedIn URL
    df = df[df["linkedin_url"].str.startswith("http")]

    if df.empty:
        raise ValueError(
            "No valid contacts found in this file. "
            "The CSV was parsed but contained no rows with a LinkedIn profile URL. "
            "Please check you exported the right file from LinkedIn."
        )

    return df.to_dict(orient="records")
