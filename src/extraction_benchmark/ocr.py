from pathlib import Path
from typing import Callable, Any, List
from loguru import logger
import asyncio


def move_if_not_yet_moved(src: Path, dest: Path) -> Path:
  """Move a file to the given destination if it's not there yet

  Args:
    src(Path): the original path to the file
    dest(Path): the destination to which the file should be moved
  Returns:
    dest(Path): the destination location
  Raises:
    Exception: if the file is neither at the source nor destination.
  """
  if src.exists():
    logger.debug(f"Moving {src} -> {dest}")
    return src.rename(dest)
  elif dest.exists():
    return dest
  else:
    raise Exception(f"Missing file: {src}")


async def extract_image_and_ocr(
  pdf_file: Path, ocr: Callable[[Any], List[str]], dpi: int = 200
) -> int:
  """
  Extract images and OCR text from the given PDF file using the provided OCR function.
  Writes the output to the parent folder of the PDF file.

  Args:
    pdf_file(Path): the pdf input file
    ocr(Callable): A function to be used for OCR processing. Takes a PIL.Image and
      returns a list of strings.
    dpi(int): The image resolution

  Returns:
    status(int):
      -1 = complete failure;
      0 = some errors occurred (partial success);
      1 = success
  """
  image_file = pdf_file.parent / f"{pdf_file.stem}_page_01.png"
  ocr_text_file = pdf_file.parent / f"{pdf_file.stem}_ocr.txt"

  if image_file.exists() and ocr_text_file.exists():
    logger.debug(f"Images and OCR for {pdf_file} already exist -> Skipping.")
    return 1

  logger.debug(f"Extracting images from {pdf_file}")
  try:
    from pdf2image import convert_from_bytes  # lazy load

    pdf_bytes = await asyncio.to_thread(pdf_file.read_bytes)
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
  except Exception as e:
    logger.error(f"Error converting PDF {pdf_file} to images: {e}")
    return -1

  # create a list to store the text pages
  pages_text: List[str] = []

  # error counter
  n_err = 0

  # iterate over page images
  for i, page in enumerate(pages):
    try:
      # save image to PNG file
      image_file = pdf_file.parent / f"{pdf_file.stem}_page_{(i + 1):02}.png"
      logger.debug(f"Writing PNG image: {image_file}")
      await asyncio.to_thread(page.save, image_file, "PNG")

      # process page with OCR and append to pages_text
      logger.debug("Running OCR")
      paragraphs = ocr(page)
      pages_text.append("\n".join(paragraphs))

    except Exception as e:
      logger.warning(f"Error parsing {pdf_file} page {i + 1}: {e}")
      n_err += 1

  # save OCR result to file
  logger.debug(f"Writing OCR output to {ocr_text_file}")
  try:
    await asyncio.to_thread(ocr_text_file.write_text, "\n\n".join(pages_text))
  except Exception:
    logger.error("Failed to write OCR text to disk!")
    n_err += 1

  if n_err == 0:
    return 1  # success
  elif n_err > len(pages):
    return -1  # total failure
  else:
    return 0  # partial failure


def lazy_load_ocr_callable() -> Callable[[Any], List[str]]:
  """
  Lazy-load the EasyOCR library and produce a Callable that can be used for OCR.

  This is a workaround for EasyOCR and numpy taking a long time to load,
  which slows down the script loading time otherwise.

  Returns:
    ocr(Callable): A function that applies OCR to a PIL.Image,
    returning a list of strings
  """
  from easyocr import Reader
  from numpy import array

  reader = Reader(["en"])

  def ocr(page: Any) -> List[str]:
    results = reader.readtext(array(page), detail=0, paragraph=True)
    return [str(item) for item in results if isinstance(item, str)]

  return ocr
