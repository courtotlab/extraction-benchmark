import asyncio
from pathlib import Path
import base64
import ollama
import os
from enum import Enum
import json
import re
from loguru import logger
import time
from typing import Any


class JsonQuality(str, Enum):
  """
  Describes the quality of the json in the response text.

  Also inherits from 'str' so that it json-serializes to a string.
  """

  PERFECT = "perfect"
  MARKDOWN_BLOCK = "markdown_block"
  BURIED = "buried"
  REPAIR_NEEDED = "repair_needed"
  UNPARSEABLE = "unparseable"


def extract_json_from_response(response: str) -> dict[str, Any]:
  """
  Extracts embedded JSON dictionaries from an LLM response.

  Handles JSON that may or may not be wrapped in markdown code blocks.
  Looks for content starting with '{' and ending with '}'.

  If more than one dict is found, it logs a warning and returns the
  longest dict

  Args:
      response: The raw LLM response string.

  Returns:
      A list of successfully parsed JSON dictionaries.
  """
  results = []
  json_quality = JsonQuality.UNPARSEABLE

  # First, try if the entire response is valid JSON
  try:
    parsed = json.loads(response)
    if isinstance(parsed, dict):
      results.append(parsed)
      json_quality = JsonQuality.PERFECT
  except json.JSONDecodeError:
    pass

  # Next, try to extract JSON from markdown code blocks
  if not results:
    # Pattern matches ```json, ```, or similar code fences
    code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)

    for block in code_blocks:
      block = block.strip()
      if block.startswith("{") and block.endswith("}"):
        try:
          parsed = json.loads(block)
          if isinstance(parsed, dict):
            results.append(parsed)
            json_quality = JsonQuality.MARKDOWN_BLOCK
        except json.JSONDecodeError:
          pass

  # If no valid JSON found in code blocks, search the response
  # for balanced braces
  if not results:
    # Find all potential JSON objects (balanced braces)
    json_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
    potential_jsons = re.findall(json_pattern, response, re.DOTALL)

    for candidate in potential_jsons:
      try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
          results.append(parsed)
          json_quality = JsonQuality.BURIED
      except json.JSONDecodeError:
        # Try with some common cleanup
        cleaned = _clean_json_string(candidate)
        try:
          parsed = json.loads(cleaned)
          if isinstance(parsed, dict):
            results.append(parsed)
            json_quality = JsonQuality.REPAIR_NEEDED
        except json.JSONDecodeError:
          continue

  # If there's more than one dict, find the longest one
  if len(results) > 1:
    logger.warning("More than one json dict found!")
    json_quality = JsonQuality.REPAIR_NEEDED
    # index using string dumps
    unique_results = {}
    for d in results:
      # Use json.dumps for consistent comparison
      key = json.dumps(d, sort_keys=True)
      unique_results[key] = d

    longest_key = max(unique_results.keys(), key=len)

    return dict(response=unique_results[longest_key], quality=json_quality)

  elif len(results) == 1:
    return dict(response=results[0], quality=json_quality)

  else:
    return dict(response=None, quality=json_quality)


def _clean_json_string(json_str: str) -> str:
  """Apply common cleanup operations to fix malformed JSON."""
  # Remove trailing commas before closing braces/brackets
  cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
  # Remove comments (both // and /* */)
  cleaned = re.sub(r"//.*?(\n|$)", r"\n", cleaned)
  cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
  # Strip whitespace
  cleaned = cleaned.strip()
  return cleaned


async def run_ollama(model: str, messages: list[dict], temp: float = 0.0) -> str:
  """
  run the ollama model on the given message context.

  Args:
    model(str): the name of the ollama model
    messages(list[dict]): the message context
  Returns:
    The response from the model
  """
  host = os.getenv("OLLAMA_HOST")
  with ollama.Client(host) as client:
    response = client.chat(
      model=model, messages=messages, options={"temperature": temp}
    )
  return response["message"]["content"]


async def _encode_image(path: Path) -> str:
  """
  Encode image at given location as base64 string
  """
  bytes = await asyncio.to_thread(path.read_bytes)
  return base64.b64encode(bytes).decode("utf-8")


async def run_experiment(experiment: dict, exp_param: dict, input_dir: Path) -> dict:
  """
  Run the experiment

  Assemble the message context using the prompt and the input data,
  then submit it to the remote LLM endpoint and return the response

  Args:
    experiment(dict): A dictionary with the details of this experiment,
      e.g. modality, quality, prompt type
    exp_param(dict): A dictionary with the contents of experimental_parameters.yaml,
      e.g. the location of the prompt text files

  Returns:
    result(dict): The result of the experiment: a dictionary with keys
      'response'(dict, the parsed json output), 'quality'(JsonQuality),
      'processing_time'(float), 'input_chars'(int), 'output_chars.(int)
  """

  prompt_type = experiment["prompt"]
  prompt_file = Path(exp_param["prompts"][prompt_type])
  prompt_text = await asyncio.to_thread(prompt_file.read_text)

  doc_id = str(experiment["doc_id"])

  image_data = None
  match experiment["modality"], experiment["quality"]:
    case "image", "original":
      doc_paths = [
        p for p in Path(input_dir / doc_id).glob(f"{doc_id}_original_page_*.png")
      ]
      if not doc_paths:
        raise Exception("No image files found!")
      image_data = [await _encode_image(path) for path in doc_paths]
      document_text = (
        "Extract the data from the document depicted in the attached images."
      )
    case "image", "distressed":
      doc_paths = [
        p for p in Path(input_dir / doc_id).glob(f"{doc_id}_distressed_page_*.png")
      ]
      if not doc_paths:
        raise Exception("No image files found!")
      image_data = [await _encode_image(path) for path in doc_paths]
      document_text = (
        "Extract the data from the document depicted in the attached images."
      )
    case "ocr_text", "original":
      # if the file is missing, let the error propagate naturally
      doc_path = Path(input_dir / doc_id / f"{doc_id}_original_ocr.txt")
      document_text = await asyncio.to_thread(doc_path.read_text)
    case "ocr_text", "distressed":
      doc_path = Path(input_dir / doc_id / f"{doc_id}_distressed_ocr.txt")
      document_text = await asyncio.to_thread(doc_path.read_text)
    case "raw_text", _:
      doc_path = Path(input_dir / doc_id / f"{doc_id}_original_raw.md")
      document_text = await asyncio.to_thread(doc_path.read_text)
    case _, _:
      raise Exception(
        (
          "Unsupported modality/quality combination: "
          "{experiment['modality']}/{experiment['quality']}"
        )
      )

  message: dict[str, str | list | dict] = {
    "role": "user",
    "content": (
      f"<instructions>\n{prompt_text}\n</instructions>\n\n"
      f"<document>\n{document_text}\n</document>"
    ),
  }
  if image_data:
    message.update({"images": image_data})

  # measure number of characters in input
  input_length = len(json.dumps(message))

  # figure out the runtime environment base on the model
  model = experiment["tool"]
  run_env = exp_param["tools"][model]["env"]

  # record the experiment start time
  start_time = time.time()

  # start the experiment
  match run_env:
    case "ollama":
      response = await run_ollama(model, [message])
    case _:
      raise Exception(f"Unsupported llm environment: {run_env}")

  # measure experiment duration
  elapsed = time.time() - start_time
  # measure the number of output characters
  output_length = len(response)

  result = extract_json_from_response(response)
  result["processing_time"] = elapsed
  result["input_chars"] = input_length
  result["output_chars"] = output_length

  return result
