import argparse
import asyncio
from pathlib import Path
from async_graph_data_flow import AsyncExecutor, AsyncGraph
from loguru import logger
import sys
import os
import json
from uuid import UUID
import pandas as pd
import yaml
from itertools import product

from extraction_benchmark.extraction import run_experiment
from extraction_benchmark.ocr import (
  extract_image_and_ocr,
  lazy_load_ocr_callable,
)
from extraction_benchmark.scoring import (
  mask_reference_data,
  score_response,
  fetch_experiment_result,
  summarize_score,
)


def _parse_args() -> argparse.Namespace:
  """
  Define and process the command line arguments

  Returns:
    args(Namespace): the parsed command line arguments
  """
  # script_path = Path(__file__)

  parser = argparse.ArgumentParser(
    description="Run data extractions for all parameter and input combinations"
  )
  default_input_path = Path(".") / "data" / "1_synthetic_data/"
  parser.add_argument(
    "--input",
    "-i",
    default=default_input_path,
    help=f"Path to input directory (default: {default_input_path})",
  )
  parser.add_argument(
    "--ollama-host",
    default="localhost:11437",
    help="The address of the Ollama host (including port number).",
  )
  return parser.parse_args()


def _read_json(json_path: Path) -> dict:
  """
  Parse the mockup_data reference data json file.

  Args:
    json_path(Path): path to mock_data.json

  Returns:
    reference_data(dict): the reference data
  """
  logger.info(f"Reading {json_path}")
  try:
    with open(json_path, "r") as f:
      reference_data = json.load(f)
  except Exception as e:
    logger.error(f"Unable to parse input file: {e}")
    sys.exit(1)
  if not isinstance(reference_data, dict):
    logger.error(
      f"Input file does not contain a dictionary! Found:{type(reference_data)}"
    )
    sys.exit(1)

  return reference_data


def _parse_yaml(param_path: Path) -> dict:
  """
  Parse the parameters yaml file and make sure it contains a dict

  Args:
    param_path(Path): parameter file path

  Returns:
    dict: The parameter data

  """
  logger.info(f"Reading {param_path}")
  try:
    with open(param_path, "r") as f:
      exp_params = yaml.safe_load(f)
  except Exception as e:
    logger.error(f"Unable to parse parameter file: {e}")
    sys.exit(1)
  if not isinstance(exp_params, dict):
    logger.error(f"Input file does not contain a dictionary! Found:{type(exp_params)}")
    sys.exit(1)

  return exp_params


def _parse_template_fields_csv(path: Path, templates: list[str]) -> pd.DataFrame:
  logger.info(f"Reading {path}")
  try:
    df = pd.read_csv(path)
  except Exception as e:
    logger.error(f"Unable to parse template fields file: {e}")
    sys.exit(1)

  if "labels" not in df.columns:
    logger.error("Missing row labels in template_fields file")
    sys.exit(1)
  df.set_index("labels", inplace=True)

  missing = [name for name in templates if name not in df.columns]
  if missing:
    logger.error(f"Missing templates from template_fields file: {', '.join(missing)}")
    sys.exit(1)

  return df


def _move_if_needed(src: Path, dest: Path) -> Path:
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


def cleanup_inputs(reference_data: dict, input_dir: Path) -> None:
  document_ids = [UUID(key) for key in reference_data.keys()]
  for doc_id in document_ids:
    # create dedicated sub-folder and move document files to it
    doc_subdir = input_dir / str(doc_id)
    doc_subdir.mkdir(exist_ok=True)

    _move_if_needed(
      input_dir / f"report_{doc_id}.pdf", doc_subdir / f"{doc_id}_original.pdf"
    )
    _move_if_needed(
      input_dir / f"report_{doc_id}.md", doc_subdir / f"{doc_id}_original_raw.md"
    )
    _move_if_needed(
      input_dir / f"report_{doc_id}_distressed.pdf",
      doc_subdir / f"{doc_id}_distressed.pdf",
    )


def _create_experimental_plan(reference_data: dict, exp_params: dict) -> pd.DataFrame:
  """
  Create a dataframe that lays out the experimental plan.
  Which inputs will be used with which parameters on which tools

  Args:
    reference_data(dict): The reference data from mock_data.json
    exp_params(dict): The experimental parameters data table
  Returns:
    experiments(DataFrame): the experimental plan
  """

  document_ids = [UUID(key) for key in reference_data.keys()]
  qualities = exp_params["qualities"]
  modalities = exp_params["modalities"]
  prompts = list(exp_params["prompts"].keys())
  tools = list(exp_params["tools"].keys())
  text_only_tools = [
    key for key, val in exp_params["tools"].items() if not val["vision"]
  ]

  # create a table of all valid experimental parameter combinations
  experiments = pd.DataFrame(
    [
      row
      for row in product(document_ids, qualities, modalities, prompts, tools)
      if not (row[1] == "distressed" and row[2] == "raw_text")
      and not (row[2] == "image" and (row[4] in text_only_tools))
    ],
    columns=["doc_id", "quality", "modality", "prompt", "tool"],
  )

  # add lab-template information based on document id
  lab_lookup = {val: key for key, val in exp_params["labs"].items()}
  doc2lab = {
    key: lab_lookup[reference_data[str(key)]["testing_laboratory"]]
    for key in document_ids
  }
  experiments.insert(
    1, "template", [doc2lab[doc_id] for doc_id in experiments["doc_id"]]
  )

  # add a sequential run-id
  experiments.insert(
    0, "run_id", [f"EXP{i:06}" for i in range(1, experiments.shape[0] + 1)]
  )

  return experiments


def setup_pipeline(
  experiments: pd.DataFrame,
  input_dir: Path,
  out_dir: Path,
  reference_data: dict,
  exp_params: dict,
) -> AsyncExecutor:
  """
  Build an asynchronous execution pipeline capable of running all
  stages in parallel

  """
  # Create the execution graph (the pipeline object)
  dag = AsyncGraph()

  # Layer 1: The input layer, feeds experimental parameter sets
  # into the pipeline
  async def input_layer():
    for _, exp in experiments.iterrows():
      yield exp

  dag.add_node(input_layer)

  # Layer 2: Optical character recognition and image extraction

  # lazy load the OCR dependency and prep an injectable function with it
  ocr = lazy_load_ocr_callable()

  async def ocr_layer(experiment):
    doc_id = experiment["doc_id"]
    pdf_original = input_dir / str(doc_id) / f"{doc_id}_original.pdf"
    pdf_distressed = input_dir / str(doc_id) / f"{doc_id}_distressed.pdf"
    await extract_image_and_ocr(pdf_original, ocr)
    await extract_image_and_ocr(pdf_distressed, ocr)
    yield experiment

  dag.add_node(ocr_layer)

  # Layer 3: Perform the experiment (i.e. call the LLM and record metrics)
  async def llm_layer(experiment):
    out_file = out_dir / f"{experiment['run_id']}_llm_result.json"
    if out_file.exists():
      logger.debug(f"Skipping experiment {experiment['run_id']}!")
      experiment["result"] = out_file
      yield experiment
    else:
      logger.debug(f"Running experiment {experiment['run_id']}")
      response = await run_experiment(experiment.to_dict(), exp_params, input_dir)
      # logger.debug(f"Archiving response for {experiment['run_id']}")
      with open(out_file, "w") as out:
        await asyncio.to_thread(json.dump, response, out, indent=2)
      experiment["result"] = out_file
      yield experiment

  dag.add_node(llm_layer)

  # Layer 4: Score the LLM output
  async def scoring_layer(experiment):
    run_id = experiment["run_id"]
    out_file = out_dir / f"{run_id}_scores.csv"
    if out_file.exists():
      logger.debug(f"Skipping existing scores for {run_id}")
      yield
    else:
      logger.debug(f"Scoring response for {run_id}")
      result = await fetch_experiment_result(experiment)
      response = result.get("response")
      if response:
        # filter down the reference data to the fields compatible with the
        # lab-specific document template
        expected_reference = mask_reference_data(
          reference_data[str(experiment["doc_id"])],
          dict(exp_params["template_fields"][experiment["template"]]),
        )
        scores = score_response(response, expected_reference)
        # logger.debug(f"Archiving scores for {run_id}")
        await asyncio.to_thread(scores.to_csv, out_file)
      yield

  dag.add_node(scoring_layer)

  dag.add_edge("input_layer", "ocr_layer")
  dag.add_edge("ocr_layer", "llm_layer")
  dag.add_edge("llm_layer", "scoring_layer")

  executor = AsyncExecutor(dag)
  return executor


def collect_results(experiments: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
  # Collect all the outputs in a final summary table
  out_table = []
  for i, experiment in experiments.iterrows():
    # prepare an output row template
    out_row = dict(experiment)
    # set default values for when no result exists
    out_row.update(
      dict(
        response="<failed>",
        parsing="<failed>",
        scores="<failed>",
        processing_time=None,
        input_chars=None,
        output_chars=None,
        tp=None,
        fp=None,
        fn=None,
        sensitivity=None,
        precision=None,
        f1=None,
      )
    )

    # check if successful output exists:
    run_id = experiment["run_id"]
    result_path = out_dir / f"{run_id}_llm_result.json"
    if not result_path.exists():
      continue
    out_row["response"] = str(result_path)

    # parse the result
    try:
      with open(result_path, "r") as infile:
        result = json.load(infile)
    except Exception:
      continue
    out_row["parsing"] = result.get("quality")
    out_row["processing_time"] = result.get("processing_time")
    out_row["input_chars"] = result.get("input_chars")
    out_row["output_chars"] = result.get("output_chars")

    # check if a score file exists
    score_path = out_dir / f"{run_id}_scores.csv"
    if not score_path.exists():
      continue
    out_row["scores"] = str(score_path)

    # parse the scores
    try:
      score_data = pd.read_csv(score_path)
    except Exception:
      continue
    score_summary = summarize_score(score_data)
    out_row.update(score_summary)
    # add the row to the output table
    out_table.append(out_row)

  # convert output to dataframe
  return pd.DataFrame(out_table)


def main() -> None:
  # parse command line arguments
  cli_args = _parse_args()
  input_dir = Path(cli_args.input)
  os.environ["OLLAMA_HOST"] = cli_args.ollama_host

  # read reference data file and experimental parameters
  reference_data = _read_json(input_dir / "mock_data.json")
  exp_params = _parse_yaml(input_dir.parent / "experiment_parameters.yaml")
  template_fields = _parse_template_fields_csv(
    input_dir.parent / "template_fields.csv", list(exp_params["labs"].keys())
  )
  exp_params["template_fields"] = template_fields

  # Cleanup the input data files
  cleanup_inputs(reference_data, input_dir)

  # create experimental plan table
  experiments = _create_experimental_plan(reference_data, exp_params)

  # export experimental plan to CSV
  logger.info("Writing experimental plan")
  out_dir = input_dir.parent / "2_llm_runs"
  out_dir.mkdir(exist_ok=True)
  experiments.to_csv(out_dir / "llm_runs.csv")

  # Set up an asynchronous execution pipeline to run all analysis layers in parallel
  logger.info("Starting pipeline")
  pipeline = setup_pipeline(experiments, input_dir, out_dir, reference_data, exp_params)
  pipeline.execute()

  # process any errors that may have occurred
  if pipeline.exceptions is not None:
    for node, es in pipeline.exceptions.items():
      if es:
        messages = [f" - {e}" for e in es[:10]]
        logger.error(f"{node} had {len(es)} exceptions:\n{'\n'.join(messages)}")

  # collect the outputs and write to file
  out_df = collect_results(experiments, out_dir)
  out_file = out_dir / "results.csv"
  logger.success(f"Writing overall result table to {out_file}")
  out_df.to_csv(out_file)


if __name__ == "__main__":
  main()
