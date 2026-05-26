import json
import asyncio
from copy import deepcopy
import pandas as pd
from typing import Any
import re


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


def normalize_string(value: str) -> str:
  """
  Normalize strings for evaluation.

  Apply a set of normaliztion rules:
  1. All upper case
  2. Remove transcript versions
  3. VUS -> Variant of uncertain significance
  4. Remove HH:MM:SS from dates

  Args:
    value(str): An input string
  Return:
    value(str): The normalized string
  """
  value = value.strip().upper()
  rules = [
    r"NM_\d+(\.\d+)?",  # remove version numbers from transcripts
    r"UNCERTAIN (CLINICAL )?SIGNIFICANCE (\(VUS\))?",  # normalize VUS label
    r"^\d{4}-\d{2}-\d{2}(.*)?",  # remove hours, minutes, seconds from dates
  ]
  for rule in rules:
    value = _remove_group_matches(value, rule)
  return value


def _remove_group_matches(value: str, rx: str) -> str:
  """
  Removes text captured by regex groups.

  Example: _remove_group_matches("ABCDE","B(C)?D") == "ABDE"

  Args:
      value: The string to process
      rx: The regular expression string containing groups

  Returns:
      String with group matches removed, or original string if
      no groups exist or no match found.
  """
  pattern = re.compile(rx)

  # Return unchanged if no groups in regex
  if pattern.groups == 0:
    return value

  # Return unchanged if pattern doesn't match
  if not pattern.search(value):
    return value

  result_parts = []
  last_end = 0

  for match in pattern.finditer(value):
    # Add text between previous match and current match
    result_parts.append(value[last_end : match.start()])

    # Collect spans of all participating groups (absolute positions)
    spans = []
    for i in range(1, pattern.groups + 1):
      start = match.start(i)
      if start != -1:  # Group participated in this match
        spans.append((start, match.end(i)))

    if not spans:
      # Groups exist but didn't capture anything (e.g., optional groups)
      result_parts.append(match.group(0))
    else:
      # Sort and merge overlapping/adjacent spans to handle nested groups
      spans.sort()
      merged = [spans[0]]
      for current in spans[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlap detected
          merged[-1] = (last[0], max(last[1], current[1]))
        else:
          merged.append(current)

      # Reconstruct the match with group contents removed
      match_start = match.start(0)
      new_parts = []
      current_pos = match_start

      for start, end in merged:
        # Keep text between previous position and group start
        new_parts.append(value[current_pos:start])
        current_pos = end

      # Keep text after last group until end of match
      new_parts.append(value[current_pos : match.end(0)])
      result_parts.append("".join(new_parts))

    last_end = match.end(0)

  # Add remaining text after last match
  result_parts.append(value[last_end:])
  return "".join(result_parts)


def _mask_field_value(value: Any, field_state: str) -> Any:
  """
  Apply a masking operation to a field value based on a field state.

  If the state is "unavailable" (not used in the template), set the value to "".
  If the state is "no_eval" (not to be evaluated), se the value to "<no_eval>"
  """
  match field_state:
    case "direct" | "indirect":
      return value
    case "unavailable":
      return ""
    case "no_eval":
      return "<no_eval>"
    case other_state:
      raise Exception(f"Unsupported field state: {other_state}")


def mask_reference_data(reference_data: dict, field_states: dict[str, str]) -> dict:
  """
  Apply allowed fields mask to reference data (based on pdf template)
  """
  # make sure we don't delete anything from the original reference data
  out = deepcopy(reference_data)

  # start by removing fields that are not requested in the extraction prompt
  # namely, num_variants, mega_hgvs, mafan and mafac
  del out["num_variants"]
  if isinstance(out["variants"], list):
    for var_dict in out["variants"]:
      del var_dict["mega_hgvs"]
      del var_dict["mafan"]
      del var_dict["mafac"]
      var_dict["maf"] = var_dict["mafaf"]
      del var_dict["mafaf"]

  # correct the value of dates to their actual document interpolations
  # out["date_collected"] = out["date_collected"].split(" ")[0]
  # out["date_received"] = out["date_received"].split(" ")[0]
  # out["date_verified"] = out["date_verified"].split(" ")[0]

  for field_name, value in out.items():
    match field_name, value, field_states[field_name]:
      case "variants", list(), _:
        for sub_dict in value:
          if isinstance(sub_dict, dict):
            for sub_field_name in sub_dict.keys():
              sub_dict[sub_field_name] = _mask_field_value(
                sub_dict[sub_field_name], field_states[sub_field_name]
              )
      # if tested_genes is no_eval, that means tested_genes as a whole should be excluded
      case "tested_genes", dict(), "no_eval":
        out[field_name] = _mask_field_value(out[field_name], field_states[field_name])
      case "tested_genes", dict(), _:
        for sub_dict in value.values():
          if isinstance(sub_dict, dict):
            for sub_field_name in sub_dict.keys():
              sub_dict[sub_field_name] = _mask_field_value(
                sub_dict[sub_field_name], field_states[sub_field_name]
              )
      case _, _, _:
        out[field_name] = _mask_field_value(out[field_name], field_states[field_name])
  return out


def score_response(response: dict, reference: dict) -> pd.DataFrame:
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
      expected = (
        "<collection>" if key in ["tested_genes", "variants"] else reference[key]
      )
      scores.append(dict(ref=key, expected=expected, found=None, tp=0, fp=0, fn=1))

  # next, check if the values match
  for key, val in response.items():
    if key in reference and reference[key] != "<no_eval>":
      match key, val:
        case "variants", list():
          # check if the entire list should be omitted from evaluation
          skip_all = all(v == "<no_eval>" for d in reference[key] for v in d.values())
          if not skip_all:
            scores.extend(score_variant_list(val, reference[key]))
        case "tested_genes", dict():
          # check if the entire list should be omitted from evaluation
          skip_all = all(
            v == "<no_eval>" for d in reference[key].values() for v in d.values()
          )
          if not skip_all:
            scores.extend(score_genes_dict(val, reference[key]))
        case (_, list()) | (_, dict()):
          # a list or dict with a different name, not allowed!
          scores.append(
            dict(
              ref=key, expected=reference[key], found=json.dumps(val), tp=0, fp=1, fn=0
            )
          )
        case (
          ("report_type", _)
          | ("testing_context", _)
          | ("sequencing_scope", _)
          | ("ordering_clinic", _)
          | ("testing_laboratory", _)
          | ("analysis_type", _)
          | ("sample_type", _)
          | ("analysis_type", _)
        ):
          scores.append(score_strings(key, str(val), str(reference[key]), exact=False))
        case _, _:
          scores.append(score_strings(key, str(val), str(reference[key])))

  return pd.DataFrame(scores)


def score_strings(field_name: str, found: str, expected: str, exact=True) -> dict:
  found = normalize_string(found)
  expected = normalize_string(expected)
  if found == expected:
    return dict(ref=field_name, expected=expected, found=found, tp=1, fp=0, fn=0)
  elif expected and not found:
    return dict(ref=field_name, expected=expected, found="", tp=0, fp=0, fn=1)
  elif found and not expected:
    return dict(ref=field_name, expected="", found=found, tp=0, fp=1, fn=0)
  elif exact:
    return dict(ref=field_name, expected=expected, found=found, tp=0, fp=1, fn=1)
  else:
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
  return f"{variant.get('gene_symbol')}:{variant.get('hgvsc')}"


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

  # if there reference omits the list entirely, score it as a FP
  if not isinstance(expected, list):
    score_entries.append(
      dict(ref="variants", expected=None, found="<list>", tp=0, fp=1, fn=0)
    )
    return score_entries

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
      elif exp_var[key] != "<no_eval>":
        score_entries.append(score_strings(key, str(found_var[key]), str(exp_var[key])))

  return score_entries


def score_genes_dict(found: dict, expected: dict) -> list[dict]:
  score_entries: list[dict] = []

  # if the reference doesn't have any, then score FP
  if not isinstance(expected, dict):
    score_entries.append(
      dict(ref="tested_genes", expected=None, found="<dict>", tp=0, fp=1, fn=0)
    )
    return score_entries

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
  for ek in exp_keys:
    if exp2found[ek] is None:
      score_entries.append(
        dict(
          ref=f"tested_genes.{ek}",
          expected=ek,
          found=None,
          tp=0,
          fp=0,
          fn=1,
        )
      )

  # Check for surplus or matching keys and report them
  for fk in found_keys:
    ek = found2exp[fk]
    if ek is None:
      score_entries.append(
        dict(
          ref=f"tested_genes.{fk}",
          expected=None,
          found=fk,
          tp=0,
          fp=1,
          fn=0,
        )
      )
      continue

    # otherwise it's a match
    exp_gene = expected[ek]
    found_gene = found[fk]

    # check for missing keys (FN)
    for key in exp_gene.keys():
      if key not in exp_gene:
        score_entries.append(
          dict(
            ref=f"tested_genes.{ek}.{key}", expected=key, found=None, tp=0, fp=0, fn=1
          )
        )
    # process remaining keys (matching or surplus)
    # FIXME: found_gene 'int' object has no attribute 'keys'
    if isinstance(found_gene, dict):
      for key in found_gene.keys():
        if key not in exp_keys:
          score_entries.append(
            dict(
              ref=f"tested_genes.{ek}.{key}", expected=None, found=key, tp=0, fp=1, fn=0
            )
          )
        elif exp_gene[key] != "<no_eval>":
          score_entries.append(
            score_strings(key, str(found_gene[key]), str(exp_gene[key]))
          )
    else:
      score_entries.append(
        dict(
          ref=f"tested_genes.{ek}",
          expected="<dict>",
          found=found_gene,
          tp=0,
          fp=1,
          fn=1,
        )
      )

  return score_entries


def summarize_score(score_table: pd.DataFrame) -> dict:
  tpsum = score_table["tp"].sum()
  fpsum = score_table["fp"].sum()
  fnsum = score_table["fn"].sum()
  recall = tpsum / (tpsum + fnsum) if tpsum > 0 else 0
  precision = tpsum / (tpsum + fpsum) if tpsum > 0 else 0
  pc = 1e-9  # pseudocount to avoid division by zero errors
  summary = dict(
    tp=tpsum,
    fp=fpsum,
    fn=fnsum,
    recall=recall,
    precision=precision,
    f1=2.0 / ((1.0 + pc) / (recall + pc) + (1.0 + pc) / (precision + pc)),
  )
  return summary


async def fetch_experiment_result(experiment: dict) -> dict:
  # If an error occurs, we just propagate up.
  with open(experiment["result"], "r") as file:
    result = await asyncio.to_thread(json.load, file)
  return result
