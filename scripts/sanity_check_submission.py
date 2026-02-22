import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image


REQUIRED_KEYS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check submission archive")
    parser.add_argument("--zip", type=str, default="submission.zip")
    return parser.parse_args()


def validate_output(output: dict) -> None:
    if not isinstance(output, dict):
        raise TypeError("Prediction output must be a dictionary.")

    for key in REQUIRED_KEYS:
        if key not in output:
            raise KeyError(f"Missing key in prediction output: {key}")

        value = output[key]
        if not isinstance(value, dict):
            raise TypeError(f"{key} output must be a dictionary with mu/sigma.")

        if "mu" not in value or "sigma" not in value:
            raise KeyError(f"{key} must contain 'mu' and 'sigma'.")

        mu = float(value["mu"])
        sigma = float(value["sigma"])

        if sigma <= 0.0:
            raise ValueError(f"{key}.sigma must be > 0. Got {sigma}.")

        _ = mu  # conversion check only


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Submission archive not found: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="submission_sanity_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import model; m=model.Model(); m.load(); print('loaded')",
            ],
            cwd=tmp_path,
            check=True,
        )

        sys.path.insert(0, str(tmp_path))
        model_module = importlib.import_module("model")
        model = model_module.Model()
        model.load()

        image = Image.new("RGB", (256, 256), color=(120, 90, 60))
        dummy_event = [
            {
                "relative_img": image,
                "colorpicker_img": image,
                "scalebar_img": image,
                "scientificName": "Unknown beetle",
                "domainID": 1,
            },
            {
                "relative_img": image,
                "colorpicker_img": image,
                "scalebar_img": image,
                "scientificName": "Unknown beetle",
                "domainID": 1,
            },
        ]

        output = model.predict(dummy_event)
        validate_output(output)

        print("predict output:")
        print(json.dumps(output, indent=2))
        print("Sanity check passed.")


if __name__ == "__main__":
    main()
