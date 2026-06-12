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
from datetime import datetime

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
  default_resp_path = Path(".") / "data" / "2_llm_runs/"
  default_score_path = Path(".") / "data" / "3_scores/"
  default_param_file_path = Path(".") / "data" / "experiment_parameters.yaml"
  default_fields_file_path = Path(".") / "data" / "template_fields.csv"
  default_ollama = "localhost:11437"
  parser.add_argument(
    "--input",
    "-i",
    default=default_input_path,
    help=f"Input directory (default: {default_input_path}).",
  )
  parser.add_argument(
    "--responses",
    "-r",
    default=default_resp_path,
    help=f"Output directory for LLM responses (default: {default_resp_path}).",
  )
  parser.add_argument(
    "--scores",
    "-s",
    default=default_score_path,
    help=f"Output directory for scores (default: {default_score_path}).",
  )
  parser.add_argument(
    "--parameters-file",
    "-p",
    default=default_param_file_path,
    help=(
      f"Path to experimental parameters JSON file (default: {default_param_file_path})"
    ),
  )
  parser.add_argument(
    "--templates-file",
    "-t",
    default=default_fields_file_path,
    help=(
      "Path to the CSV file holding template field classifications "
      f"(default: {default_fields_file_path})"
    ),
  )
  parser.add_argument(
    "--ollama-host",
    default=default_ollama,
    help=f"The host and port for the Ollama server (default: {default_ollama}).",
  )
  parser.add_argument(
    "--skip-cleanup", help="Skip the input file cleaning process", action="store_true"
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


def _parse_template_fields_csv(path: Path, templates: list[str] | None) -> pd.DataFrame:
  """
  Parse the template fields CSV file.

  This file describes which LaTeX templates use which reference data fields.
  Column names are expected to be template names (corresponding to 'labs' in
  the experimental parameters file), while row names are expected to correspond
  to the possible reference json field names (i.e. dict keys). The value in each
  cell should be "direct" (for directly used); "indirect" (for indirectly used, such
  that the value is logically inferrable); "unavailable" (not used in this
  template and thus expected to be empty); or "no_eval" (indicating that the template
  contains hard-coded content that misrepresents the reference data and should
  thus be excluded from evaluation)

  Args:
    path(Path): The path to the csv file
    templates(list[str]): the list of allowed template names (from the experimental
      parameters file)

  Returns:
    df(DataFrame): A data frame of the csv table
  """
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

  if templates:
    missing = [name for name in templates if name not in df.columns]
    if missing:
      logger.error(f"Missing templates from template_fields file: {', '.join(missing)}")
      sys.exit(1)

  return df


def _move_if_needed(src: Path, dest: Path) -> Path:
  """Move a file to the given destination if it's not there yet.

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
  """
  Iterate over all reference data and organize the associated input files.

  For each reference document it checks if the associated pdf and md files are
  already in a dedicated subfolder, if not, it creates the folder and moves
  the documents into it.

  Propagates any exceptions arising at lower levels (e.g. if files are
  missing or can't be read)

  Args:
    reference_data(dict): The reference data set. Keys are expected to
      be the UUIDs representing a reference document
    input_dir(Path): The directory where the documents are located
  """
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
  if "labs" in exp_params:
    lab_lookup = {val: key for key, val in exp_params["labs"].items()}
    doc2lab = {
      key: lab_lookup[reference_data[str(key)]["testing_laboratory"]]
      for key in document_ids
    }
    experiments.insert(
      1, "template", [doc2lab[doc_id] for doc_id in experiments["doc_id"]]
    )
  else:
    # This might be the case if we're using other external datasets with no templates
    logger.warning("No labs listed. Inserting default template!")
    experiments.insert(1, "template", "default")

  # sort by model to reduce ollama model load-time overhead
  experiments.sort_values(by=["tool", "modality", "quality", "prompt"], inplace=True)

  # add a sequential run-id
  experiments.insert(
    0, "run_id", [f"EXP{i:06}" for i in range(1, experiments.shape[0] + 1)]
  )

  return experiments


def setup_pipeline(
  experiments: pd.DataFrame,
  input_dir: Path,
  llm_out_dir: Path,
  scores_out_dir: Path,
  reference_data: dict,
  exp_params: dict,
) -> AsyncExecutor:
  """
  Build an asynchronous execution pipeline capable of running all
  stages in parallel

  Args:
    experiments(DataFrame): The experiments table
    input_dir(Path): The input directory
    llm_out_dir(Path): The llm output directory
    scores_out_dir(Path): The scoring output directory
    reference_data(dict): The reference data dictionary (from mock_data.json)
    exp_params(dict): The experimental parameters (from experiment_parameters.yaml)

  Returns:
    pipeline(AsyncExecutor): The pipeline object
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
    if "original" in exp_params["qualities"]:
      pdf_original = input_dir / str(doc_id) / f"{doc_id}_original.pdf"
      await extract_image_and_ocr(pdf_original, ocr)
    if "distressed" in exp_params["qualities"]:
      pdf_distressed = input_dir / str(doc_id) / f"{doc_id}_distressed.pdf"
      await extract_image_and_ocr(pdf_distressed, ocr)
    yield experiment

  dag.add_node(ocr_layer)

  # Layer 3: Perform the experiment (i.e. call the LLM and record metrics)
  async def llm_layer(experiment):
    out_file = llm_out_dir / f"{experiment['run_id']}_llm_result.json"
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
    out_file = scores_out_dir / f"{run_id}_scores.csv"
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


def collect_results(
  experiments: pd.DataFrame, llm_out_dir: Path, scores_out_dir: Path
) -> pd.DataFrame:
  """
  Collect all the results and compile them in a table with the following
  columns: run_id, doc_id, template, quality (original/distressed),
  modality (text/image), prompt (zero-shot/one-shot), tool (llm model),
  response (path to llm response), parsing_quality
  (perfect/markdown/repair_needed/failed),
  scores (path to score file), processing_time (llm request duration),
  input_tokens, output_tokens, tp (true positives), fp (false positives),
  fn (false negatives), recall, precision, f1 (F1-score)

  Args:
    experiments(DataFrame): The experiments table
    llm_out_dir(Path): The llm output directory
    scores_out_dir(Path): The score output directory

  Returns:
    results(DataFrame): The compiled results
  """
  # Collect all the outputs in a final summary table
  out_table = []
  for i, experiment in experiments.iterrows():
    # prepare an output row template
    out_row = dict(experiment)
    # set default values for when no result exists
    out_row.update(
      dict(
        response="<failed>",
        parsing_quality="<failed>",
        scores="<failed>",
        processing_time=None,
        input_tokens=None,
        output_tokens=None,
        tp=None,
        fp=None,
        fn=None,
        recall=None,
        precision=None,
        f1=None,
      )
    )

    # check if successful output exists:
    run_id = experiment["run_id"]
    result_path = llm_out_dir / f"{run_id}_llm_result.json"
    if not result_path.exists():
      # add the row to the output table
      out_table.append(out_row)
      continue
    out_row["response"] = str(result_path)

    # parse the result
    try:
      with open(result_path, "r") as infile:
        result = json.load(infile)
    except Exception:
      # add the row to the output table
      out_table.append(out_row)
      continue
    out_row["parsing_quality"] = result.get("quality")
    out_row["processing_time"] = result.get("processing_time")
    out_row["input_tokens"] = result.get("input_tokens")
    out_row["output_tokens"] = result.get("output_tokens")

    # check if a score file exists
    score_path = scores_out_dir / f"{run_id}_scores.csv"
    if not score_path.exists():
      # add the row to the output table
      out_table.append(out_row)
      continue
    out_row["scores"] = str(score_path)

    # parse the scores
    try:
      score_data = pd.read_csv(score_path)
    except Exception:
      # add the row to the output table
      out_table.append(out_row)
      continue
    score_summary = summarize_score(score_data)
    out_row.update(score_summary)
    # add the row to the output table
    out_table.append(out_row)

  # convert output to dataframe
  return pd.DataFrame(out_table)


def main() -> None:
  # set up log file
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  log_file = Path(".") / f"benchmark_{timestamp}.log"
  logger.add(log_file, level="DEBUG")

  # parse command line arguments
  cli_args = _parse_args()
  input_dir = Path(cli_args.input)
  llm_out_dir = Path(cli_args.responses)
  scores_out_dir = Path(cli_args.scores)
  param_path = Path(cli_args.parameters_file)
  tmpl_fields_path = Path(cli_args.templates_file)
  os.environ["OLLAMA_HOST"] = cli_args.ollama_host

  # read reference data file and experimental parameters
  reference_data = _read_json(input_dir / "mock_data.json")
  exp_params = _parse_yaml(param_path)
  templ_names = list(exp_params["labs"].keys()) if "labs" in exp_params else None
  template_fields = _parse_template_fields_csv(tmpl_fields_path, templ_names)
  exp_params["template_fields"] = template_fields

  # if any experiments require OpenAI, check that the API key is available
  if any(
    tool_config["env"] == "openai" for tool_config in exp_params["tools"].values()
  ):
    if not os.environ["OPENAI_API_KEY"]:
      logger.error("OPENAI_API_KEY variable is required, but not defined!")
      sys.exit(1)

  # Cleanup the input data files (unless user asked to skip)
  if not cli_args.skip_cleanup:
    cleanup_inputs(reference_data, input_dir)

  # load or create experimental plan table
  llm_out_dir.mkdir(exist_ok=True)
  experiments_file = llm_out_dir / "llm_runs.csv"
  if experiments_file.exists():
    logger.info(f"Loading experiment plan from existing file: {experiments_file}")
    experiments = pd.read_csv(experiments_file)
  else:
    # create a new plan plan
    experiments = _create_experimental_plan(reference_data, exp_params)
    # export experimental plan to CSV
    logger.info("Writing experimental plan")
    experiments.to_csv(experiments_file)

  scores_out_dir.mkdir(exist_ok=True)

  # Set up an asynchronous execution pipeline to run all analysis layers in parallel
  logger.info("Starting pipeline")
  pipeline = setup_pipeline(
    experiments, input_dir, llm_out_dir, scores_out_dir, reference_data, exp_params
  )
  pipeline.execute()

  # process any errors that may have occurred
  if pipeline.exceptions is not None:
    for node, es in pipeline.exceptions.items():
      if es:
        messages = [f" - {e}" for e in es[:10]]
        logger.error(f"{node} had {len(es)} exceptions:\n{'\n'.join(messages)}")

  # collect the outputs and write to file
  out_df = collect_results(experiments, llm_out_dir, scores_out_dir)
  out_file = scores_out_dir / "results.csv"
  logger.success(f"Writing overall result table to {out_file}")
  out_df.to_csv(out_file)


if __name__ == "__main__":
  main()
