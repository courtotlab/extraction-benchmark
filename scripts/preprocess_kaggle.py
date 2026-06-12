import json
import subprocess
import uuid
from pathlib import Path

from extraction_benchmark.ocr import lazy_load_ocr_callable
from PIL import Image

import pandas as pd

ocr = lazy_load_ocr_callable()

indir = Path("data") / "kaggle"

in_subdir = indir / "discharge_summaries"
# in_subdir = indir / "bills"
outdir = Path("data") / "discharge"
# outdir = Path("data") / "bills"

csv_path = indir / "discharge_summaries_ground_truth.csv"
# csv_path = indir / "medical_bills_ground_truth.csv"
data = pd.read_csv(csv_path)

uuids = [str(uuid.uuid4()) for _ in range(len(data))]

parsed = [json.loads(x) for x in data.iloc[:, 2]]
json_data = dict(zip(uuids, parsed))

outdir.mkdir(exist_ok=True)
json_path = outdir / "mock_data.json"
with open(json_path, "w") as f:
  json.dump(json_data, f, indent=2)


def convert_image(infile, doc_id, target_dir):
  outfile = target_dir / f"{doc_id}_original_page_01.png"
  text_out_file = target_dir / f"{doc_id}_original_ocr.txt"
  print(f"Converting {infile} ...")
  # Use imagemagick to convert the file to png
  result = subprocess.run(["magick", infile, outfile], capture_output=True)
  if result.returncode != 0:
    print("Image conversion failed!")
  # load the image and perform OCR
  page = Image.open(infile)
  paragraphs = ocr(page)
  text = "\n".join(paragraphs)
  text_out_file.write_text(text)


for i in range(len(data)):
  input_img_path = in_subdir / str(data.iloc[i, 0])
  img_outdir = outdir / uuids[i]
  img_outdir.mkdir(exist_ok=True)
  convert_image(input_img_path, uuids[i], img_outdir)
