#!/usr/bin/env python3
"""
gen_descriptions.py — GPT-4o natural-language descriptions for each typhoon
record. This is offline preprocessing for TyphoFormer-C; it necessarily runs
in Python (OpenAI API) — the C code consumes the resulting embeddings, not the
model. Set OPENAI_API_KEY in the environment.

Usage:
    OPENAI_API_KEY=... python tools/gen_descriptions.py <in.csv> <out.csv>
    # e.g.  ... ../HURDAT_2new_3000.csv ../HURDAT_2new_3000_with_descriptions.csv
"""
import os
import sys
import time

import pandas as pd
from openai import OpenAI


def build_prompt(row):
    record = "\n".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v))
    return (
        "You are a senior meteorologist. Given the structured typhoon record "
        "below, write a concise (3-6 sentence) natural-language description of "
        "the storm's state at that time: position, intensity (max wind, central "
        "pressure), the extent of the 34/50/64-kt wind radii, and the likely "
        "phase (formation, intensification, peak, weakening, or landfall). "
        "Skip any missing values.\n\n"
        f"Record:\n{record}\n\nDescription:"
    )


def main(in_csv, out_csv):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    df = pd.read_csv(in_csv)
    if "description" not in df.columns:
        df["description"] = ""
    for i, row in df.iterrows():
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": build_prompt(row)}],
                temperature=0.3, max_tokens=300,
            )
            df.at[i, "description"] = resp.choices[0].message.content.strip()
        except Exception as e:                       # noqa: BLE001
            print(f"row {i}: error {e}")
            df.at[i, "description"] = "ERROR"
        if (i + 1) % 10 == 0:
            df.to_csv(out_csv, index=False)
            print(f"progress {i + 1}/{len(df)}")
        time.sleep(1)                                # gentle rate control
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
