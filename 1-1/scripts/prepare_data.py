"""Download UCI data and convert XLSX using only Python's standard library.

The NumPy fixtures are synthetic, reproducible images, NOT product photographs.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def prepare():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "online_retail.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("Online Retail.xlsx"))) as book:
            strings_root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in strings_root]
            with book.open("xl/worksheets/sheet1.xml") as sheet, (raw / "transactions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.writer(output)
                writer.writerow(["order_id", "product_id", "product_name", "quantity",
                                 "order_date", "price", "customer_id", "country"])
                count = 0
                for _, row in ET.iterparse(sheet, events=("end",)):
                    if row.tag != NS + "row":
                        continue
                    if row.attrib["r"] == "1":
                        row.clear()
                        continue
                    values = [""] * 8
                    for cell in row:
                        col = cell.attrib["r"].rstrip("0123456789")
                        index = ord(col) - ord("A")
                        value = cell.find(NS + "v")
                        if index < 8 and value is not None:
                            text = value.text or ""
                            values[index] = strings[int(text)] if cell.attrib.get("t") == "s" else text
                    writer.writerow(values)
                    count += 1
                    row.clear()
    df = pd.read_csv(raw / "transactions.csv", dtype={"product_id": str})
    products = np.sort(df.product_id.unique().astype(str))
    rng = np.random.default_rng(20261025)
    # Batch construction and image statistics contain no pixel/image loops.
    colors = rng.integers(30, 220, size=(len(products), 1, 1, 3))
    noise = rng.normal(0, 25, size=(len(products), 16, 16, 3))
    images = np.clip(colors + noise, 0, 255).astype(np.uint8)
    np.savez_compressed(ROOT / "data/image_fixtures.npz", product_ids=products, images=images)
    manifest = {
        "source": URL, "citation": "Chen, D. (2015). Online Retail. DOI: 10.24432/C5BW33",
        "license": "CC BY 4.0", "raw_rows": count, "raw_columns": 8,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256((raw / "transactions.csv").read_bytes()).hexdigest(),
        "image_origin": "Synthetic NumPy fixtures; independent of prices and sales; not photographs",
        "image_seed": 20261025, "image_shape": list(images.shape),
        "date_encoding": "Excel serial days, origin 1899-12-30",
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
