import json
import asyncio
from copy import deepcopy
import pandas as pd


# def filter_reference_data(reference_data: dict, allowed_fields: list[str]) -> dict:
#   out: dict[str, Any] = {}
#   for field_name, value in reference_data.items():
#     if field_name in allowed_fields:
#       if isinstance(value, dict):
#         out[field_name] = value.copy()
#       elif isinstance(value, list):
#         out[field_name] = [
#           filter_reference_data(sub_dict, allowed_fields) for sub_dict in value
#         ]
#       else:
#         out[field_name] = value
#   return out

import numpy as np


def _align_strings(
  ref_str, found_str, match_score=1, mismatch_score=1, gap_penalty=1
) -> tuple[int, int, int, int]:
  """
  Align two strings, character by character, using the
  Needleman-Wunsch algorithm, and return alignment statistics.

  Args:
    ref_str(str): The reference string
    found_str(str): The string to align against the reference
    match_score(int): Score for character match (default: 1)
    mismatch_score(int): Penalty for character mismatch (default: 1)
    gap_penalty(int): Penalty for gap (insertion/deletion) (default: 1)

  Returns:
    tuple: Tuple containing matches, mismatches, insertions, and deletions
  """
  nx = len(ref_str)
  ny = len(found_str)

  # Initialize scoring matrix
  F = np.zeros((nx + 1, ny + 1))

  # Initialize gap penalties for first row and column
  for i in range(nx + 1):
    F[i, 0] = -i * gap_penalty
  for j in range(ny + 1):
    F[0, j] = -j * gap_penalty

  # Fill the scoring matrix
  for i in range(1, nx + 1):
    for j in range(1, ny + 1):
      # Score for match/mismatch
      if ref_str[i - 1] == found_str[j - 1]:
        score_match = F[i - 1, j - 1] + match_score
      else:
        score_match = F[i - 1, j - 1] - mismatch_score

      # Score for gap in found_str (deletion from ref_str)
      score_delete = F[i - 1, j] - gap_penalty

      # Score for gap in ref_str (insertion into ref_str)
      score_insert = F[i, j - 1] - gap_penalty

      F[i, j] = max(score_match, score_delete, score_insert)

  # Traceback to reconstruct alignment
  aligned_ref: list[str] = []
  aligned_found: list[str] = []

  i, j = nx, ny
  while i > 0 or j > 0:
    if i > 0 and j > 0:
      # Check if we came from diagonal (match/mismatch)
      if ref_str[i - 1] == found_str[j - 1]:
        match_val = F[i - 1, j - 1] + match_score
      else:
        match_val = F[i - 1, j - 1] - mismatch_score

      if F[i, j] == match_val:
        aligned_ref.append(ref_str[i - 1])
        aligned_found.append(found_str[j - 1])
        i -= 1
        j -= 1
        continue

    # Check if we came from above (deletion from ref_str)
    if i > 0 and F[i, j] == F[i - 1, j] - gap_penalty:
      aligned_ref.append(ref_str[i - 1])
      aligned_found.append("-")
      i -= 1
    # Otherwise, came from left (insertion into ref_str)
    elif j > 0:
      aligned_ref.append("-")
      aligned_found.append(found_str[j - 1])
      j -= 1

  # Reverse to get correct order
  aligned_ref = aligned_ref[::-1]
  aligned_found = aligned_found[::-1]

  # Count statistics
  matches = 0
  mismatches = 0
  insertions = 0  # gaps in ref_str (characters added to ref_str)
  deletions = 0  # gaps in found_str (characters removed from ref_str)

  for a, b in zip(aligned_ref, aligned_found):
    if a == b:
      matches += 1
    elif a == "-":
      insertions += 1  # Character in found_str but not in ref_str
    elif b == "-":
      deletions += 1  # Character in ref_str but not in found_str
    else:
      mismatches += 1  # Different characters

  return (matches, mismatches, insertions, deletions)


def mask_reference_data(reference_data: dict, allowed_fields: list[str]) -> dict:
  """
  Apply allowed fields mask to reference data (based on pdf template)
  """
  # make sure we don't delete anything from the original reference data
  out = deepcopy(reference_data)
  for field_name, value in out.items():
    if field_name not in allowed_fields:
      out[field_name] = ""
    elif isinstance(value, list):
      for sub_dict in value:
        if isinstance(sub_dict, dict):
          for sub_field_name in sub_dict.keys():
            if sub_field_name not in allowed_fields:
              sub_dict[sub_field_name] = ""
  return out


"""
"tested_genes": {       #dict of dicts
 "PSIP1": {
 "gene_symbol": "PSIP1",
"refseq_mrna": "NM_033222.5" 
},

"variants": [          #list of dicts
 {
 "gene_symbol": "JMJD1C",
"variant_id": "VCV007184459",
"chromosome": "chr10",
"hgvsg": "g.63172803A>G",
"""


async def score_response(response: dict, reference: dict) -> pd.DataFrame:
  """
  Score the correctness of the response relative to the reference data.

  Args:
    response(dict): The json dictionary containing the data the LLM
      extracted from the document.
    reference(dict): The json dictionary containing the reference data
  Returns:
    scores(DataFrame): A DataFrame listing the score for each individual field
      across the response and reference. Columns: "ref" (the json key),
      "expected" (the expected value or None if hallucination),
      "found" (the value extracted by the LLM or None if missing),
      "tp" (the true positive score [0;1]), "fp" (the false positive score),
      and "fn" (the false negative score)
  """
  scores = []
  # first, check if we have hallucinated fields
  for key in response.keys():
    if key not in reference:
      scores.append(dict(ref=None, expected=None, found=key, tp=0, fp=1, fn=0))

  # next, check if fields are missing
  for key in reference.keys():
    if key not in response:
      scores.append(
        dict(ref=key, expected=reference[key], found=None, tp=0, fp=0, fn=1)
      )

  # next, check if the values match
  for key, val in response.items():
    if key in reference:
      match key, val:
        case "variants", list():
          scores.extend(score_variant_list(val, reference[key]))
          pass
        case "tested_genes", dict():
          scores.extend(score_genes_dict(val, reference[key]))
          pass
        case (_, list()) | (_, dict()):
          # a list or dict with a different name, not allowed!
          scores.append(
            dict(
              ref=key, expected=reference[key], found=json.dumps(val), tp=0, fp=1, fn=0
            )
          )
          pass
        case _, _:
          scores.append(score_strings(key, str(val), str(reference[key])))

  return pd.DataFrame(scores)


def score_strings(field_name: str, found: str, expected: str) -> dict:
  found = found.strip()
  expected = expected.strip()
  if found == expected:
    return dict(ref=field_name, expected=expected, found=found, tp=1, fp=0, fn=0)
  elif found == "":
    return dict(ref=field_name, expected=expected, found="", tp=0, fp=0, fn=1)
  elif expected == "":
    return dict(ref=field_name, expected="", found=found, tp=0, fp=1, fn=0)

  matches, mismatches, insertions, deletions = _align_strings(
    expected.upper(), found.upper()
  )
  fp = (insertions + mismatches) / len(found)
  fn = (deletions + mismatches) / len(expected)
  tp = matches / len(expected)
  return dict(ref=field_name, expected=expected, found=found, tp=tp, fp=fp, fn=fn)


def _var_label(variant: dict) -> str:
  """
  Create an identifying label for a variant dict, e.g. BRCA1:c.123A>C.
  if gene_symbol or hgvsc are not defined, label segment will be "None"
  """
  return f"{variant.get('gene_symbol'):{variant.get('hgvsc')}}"


def _greedy_pairings(
  found_strs: list[str], expected_strs: list[str]
) -> tuple[list[int], list[int]]:
  """
  Greedily forms exclusive pairs between lists of expected and found
  strings based on a alignment-based cost function
  """

  # function to calculate assignment cost
  def _cost(ev: str, fv: str) -> int:
    if ev == fv:
      # if they are a perfect match, the cost is
      # -2 * number of character matches
      return -2 * len(ev)
    # otherwise, align them and calculate cost as
    # character mismatches + indels - matches
    m, mm, ins, dl = _align_strings(ev, fv)
    return mm + ins + dl - m

  # calculate the cost matrix
  nrow = len(found_strs)
  ncol = len(expected_strs)
  cost_mat = [[_cost(ev, fv) for ev in expected_strs] for fv in found_strs]

  # assignment trackers (-1 means unassigned)
  row2col = [-1] * nrow
  col2row = [-1] * ncol

  # greedy assignment algorithm
  while True:
    greedy_row = -1
    greedy_col = -1
    min_cost = 2**63
    for i in range(nrow):
      # if the row is already assigned, skip it
      if row2col[i] > -1:
        continue
      for j in range(ncol):
        # if the column is not yet assigned and the cost is better
        if col2row[j] < 0 and cost_mat[i][j] < min_cost:
          # then update the best cost and record the match
          min_cost = cost_mat[i][j]
          greedy_row = i
          greedy_col = j
    # if we can't make an assignment, then we're done
    if greedy_row < 0 or greedy_col < 0:
      break
    # assign the pairing
    row2col[greedy_row] = greedy_col
    col2row[greedy_col] = greedy_row

  return row2col, col2row


def score_variant_list(found: list[dict], expected: list[dict]) -> list[dict]:
  score_entries: list[dict] = []

  # greedily pair up the two variant lists based on gene symbol and HGVS
  found2exp, exp2found = _greedy_pairings(
    [_var_label(var) for var in found], [_var_label(var) for var in expected]
  )

  # if there are any missing variants, report them as FN
  missing = [i for i in range(len(expected)) if exp2found[i] == -1]
  for i in missing:
    score_entries.append(
      dict(
        ref=f"variants[{i}]",
        expected=_var_label(expected[i]),
        found=None,
        tp=0,
        fp=0,
        fn=1,
      )
    )

  # if there are surplus variants found, report them as FP
  surplus = [i for i in range(len(found)) if found2exp[i] == -1]
  for i in surplus:
    score_entries.append(
      dict(
        ref=f"variants[{i}]",
        expected=None,
        found=_var_label(found[i]),
        tp=0,
        fp=1,
        fn=0,
      )
    )

  # finally, we compare the matched variants in detail:
  for i in range(len(expected)):
    # get the mapping and and skip if there's none
    j = exp2found[i]
    if j < 0:
      continue

    # resolve the mapping to the actual variant dicts
    exp_var = expected[i]
    found_var = found[j]

    # check for missing keys (FN)
    for key in exp_var.keys():
      if key not in found_var:
        score_entries.append(
          dict(ref=f"variants[{i}].{key}", expected=key, found=None, tp=0, fp=0, fn=1)
        )
    # process remaining keys (matching or surplus)
    for key in found_var.keys():
      if key not in exp_var:
        score_entries.append(
          dict(ref=f"variants[{i}].{key}", expected=None, found=key, tp=0, fp=1, fn=0)
        )
      else:
        score_entries.append(score_strings(key, str(found_var[key]), str(exp_var[key])))

  return score_entries


def score_genes_dict(found: dict, expected: dict) -> list[dict]:
  score_entries: list[dict] = []

  found_keys = list(found.keys())
  exp_keys = list(expected.keys())
  f2e, e2f = _greedy_pairings(found_keys, exp_keys)
  found2exp = {
    fk: (exp_keys[f2e[i]] if f2e[i] > -1 else None) for i, fk in enumerate(found_keys)
  }
  exp2found = {
    ek: (found_keys[e2f[i]] if e2f[i] > -1 else None) for i, ek in enumerate(exp_keys)
  }

  # Check for missing keys and report them as FN

  # Check for surplus or matching keys and report them

  return score_entries


async def summarize_score(scores: dict) -> dict:
  return {}


async def fetch_experiment_result(experiment: dict) -> dict:
  # If an error occurs, we just propagate up.
  with open(experiment["result"], "r") as file:
    result = await asyncio.to_thread(json.load, file)
  return result
