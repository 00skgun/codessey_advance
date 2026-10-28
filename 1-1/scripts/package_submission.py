"""Verify the executed deliverables and package them without local environments."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = json.loads((ROOT / "notebooks/analysis_report.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells), "Execute the notebook first"
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    assert not any(output["output_type"] == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 6
    execution = json.loads((ROOT / "outputs/execution_summary.json").read_text(encoding="utf-8"))
    assert execution["status"] == "passed", "Record a successful --execute run first"
    assert execution["executed_cells"] == len(code_cells) and execution["error_outputs"] == 0
    assert execution["notebook_sha256"] == hashlib.sha256(
        (ROOT / "notebooks/analysis_report.ipynb").read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert execution["config_used"] == json.loads((ROOT / "outputs/config_used.json").read_text(encoding="utf-8"))
    for name, expected in execution["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        assert actual == expected, "Source changed since execution: " + name
    figures = json.loads((ROOT / "outputs/figure_manifest.json").read_text(encoding="utf-8"))
    assert len(figures) == 6
    for figure in figures:
        assert figure["title"] and figure["xlabel"] and figure["ylabel"]
        assert hashlib.sha256((ROOT / figure["path"]).read_bytes()).hexdigest() == figure["sha256"]
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "data/raw/transactions.csv").read_bytes()).hexdigest() == manifest["csv_sha256"]
    audit = json.loads((ROOT / "outputs/quality_audit.json").read_text(encoding="utf-8"))
    assert audit["raw_rows"] == 541909
    assert audit["valid_purchase_rows"] + audit["excluded_rows"] == audit["raw_rows"]
    target = ROOT / "outputs/ecommerce_submission_bundle.zip"
    folders = [ROOT / name for name in ["src", "scripts", "notebooks", "data", "outputs", "tests"]]
    files = [ROOT / name for name in ["README.md", ".gitignore", "requirements.txt", "requirements-dev.txt", "config.json"]]
    files.extend(path for folder in folders for path in folder.rglob("*") if path.is_file())
    files = sorted(path for path in files if "__pycache__" not in path.parts
                   and path.suffix not in {".pyc", ".zip"})
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
        assert "src/pipeline.py" in bundle.namelist()
        assert "notebooks/analysis_report.ipynb" in bundle.namelist()
        assert "README.md" in bundle.namelist()
    print(json.dumps({"files": len(files), "executed_code_cells": len(code_cells),
                      "embedded_charts": 6, "bundle_bytes": target.stat().st_size,
                      "bundle": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
