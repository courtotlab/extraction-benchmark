import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from typing import Union, Optional
from pathlib import Path
import asyncio
from qwen_vl_utils import process_vision_info, fetch_image


def concatenate_images_vertical(
  images: list[Image.Image], background: Union[str, tuple] = (255, 255, 255, 0)
) -> Image.Image:
  """
  Concatenates a list of PIL images vertically with left alignment.

  Args:
      images: List of PIL Image objects to concatenate
      background: Background color for padding. Can be a color name string
        (e.g., 'white') or RGB/RGBA tuple. Default is transparent.

  Returns:
      A new PIL Image with all images stacked vertically

  Raises:
      ValueError: If the images list is empty
  """
  if not images:
    raise ValueError("At least one image is required")

  # Get maximum width and total height
  max_width = max(img.width for img in images)
  total_height = sum(img.height for img in images)

  # Determine mode - use RGBA if any image has alpha, otherwise match first image
  has_alpha = any(img.mode == "RGBA" for img in images)
  mode = "RGBA" if has_alpha else images[0].mode

  # Create output image with determined mode
  if isinstance(background, str):
    result = Image.new(mode, (max_width, total_height), background)
  else:
    result = Image.new(mode, (max_width, total_height), background)

  # Paste images vertically with left alignment
  y_offset = 0
  for img in images:
    # Convert image mode if necessary for compatibility
    if img.mode != mode:
      img = img.convert(mode)

    # Paste at left edge (x=0) with current y offset
    result.paste(img, (0, y_offset))
    y_offset += img.height

  return result


class NuExtract:
  """
  A wrapper class for the NuExtract transformer model.

  Much of this code has been adapted from
  https://github.com/numindai/nuextract/blob/main/cookbooks/nuextract-2.0_inference.ipynb
  """

  instance: Optional[NuExtract] = None

  @classmethod
  def get_instance(cls):
    if not cls.instance:
      # print("New instance!")
      cls.instance = NuExtract()
    return cls.instance

  def __init__(
    self,
    model_name: str = "numind/NuExtract-2.0-2B",  # "numind/NuExtract-2.0-8B"
  ):

    if torch.backends.mps.is_available():
      self.device = "mps"
      self.torch_dtype = torch.float16
    elif torch.cuda.is_available():
      self.device = "cuda"
      self.torch_dtype = torch.bfloat16
    else:
      self.device = "cpu"
      self.torch_dtype = torch.float32

    self.model = (
      AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=self.torch_dtype,
        # attn_implementation="flash_attention_2",
        device_map="auto",
      )
      .to(self.device)
      .eval()
    )

    # You can set min_pixels and max_pixels according to your needs,
    # such as a token range of 256-1280, to balance performance and cost.
    self.processor = AutoProcessor.from_pretrained(
      model_name,
      trust_remote_code=True,
      padding_side="left",
      use_fast=True,
      min_pixels=256 * 28 * 28,
      max_pixels=1280 * 28 * 28,
    )

  def _process_all_vision_info(
    self,
    messages: Union[list[dict], list[list[dict]]],
    examples: Optional[Union[list[dict], list[list[dict]]]] = None,
  ) -> Optional[list[Image.Image]]:
    """
    Process vision information from both messages and in-context examples,
    supporting batch processing.
    Args:
        messages: List of message dictionaries (single input)
          OR list of message lists (batch input)
        examples: Optional list of example dictionaries (single input)
          OR list of example lists (batch)

    Returns:
        A flat list of all images in the correct order:
        - For single input: example images followed by message images
        - For batch input: interleaved as
          (item1 examples, item1 input, item2 examples, item2 input, etc.)
        - Returns None if no images were found
    """
    # from qwen_vl_utils import process_vision_info, fetch_image

    # Helper function to extract images from examples
    def extract_example_images(
      example_item: Union[dict, list[dict]],
    ) -> list[Image.Image]:
      if not example_item:
        return []

      # Handle both list of examples and single example
      examples_to_process = (
        example_item if isinstance(example_item, list) else [example_item]
      )
      images = []

      for example in examples_to_process:
        if (
          isinstance(example.get("input"), dict)
          and example["input"].get("type") == "image"
        ):
          images.append(fetch_image(example["input"]))

      return images

    # Normalize inputs to always be batched format
    messages_batch: list[list[dict]]
    match messages:
      case [dict(), *_]:  # list of dicts
        messages_batch = [messages]  # type: ignore
      case [[dict(), *_], *_]:  # list of list of dicts
        messages_batch = messages  # type: ignore
      case _:
        raise Exception("messages must be list of dict or list of list of dict")

    examples_batch: Optional[list[list[dict]]]
    match examples:
      case [dict(), *_]:  # list of dicts
        examples_batch = [examples]  # type: ignore
      case [[dict(), *_]]:  # list of list of dicts
        examples_batch = examples  # type: ignore
      case _:
        examples_batch = None

    # Ensure examples batch matches messages batch if provided
    if examples_batch and len(examples_batch) != len(messages_batch):
      raise ValueError("Examples batch length must match messages batch length")

    # Process all inputs, maintaining correct order
    all_images: list[Image.Image] = []
    for i, message_group in enumerate(messages_batch):
      # Get example images for this input
      if examples_batch and i < len(examples_batch):
        input_example_images = extract_example_images(examples_batch[i])
        all_images.extend(input_example_images)

      # Get message images for this input
      input_message_images = process_vision_info(message_group)[0] or []
      all_images.extend(input_message_images)

    return all_images if all_images else None

  def run(
    self,
    template: str,
    document: Union[
      str, dict[str, Union[str, Image.Image]], list[dict[str, Union[str, Image.Image]]]
    ],
    examples: Optional[
      list[dict[str, Union[str, dict[str, Union[str, Image.Image]]]]]
    ] = None,
    output_cutoff=4 * 2**10,
  ) -> list[str]:
    """
    Run the NuExtract model.

    Requires a json template string, and a document (which can be an image or text).
    Optionally accepts a list of few-shot examples.

    template = '{"names": ["string"]}'
    document = "John went to the restaurant with Mary. James went to the cinema."
    document = {"type": "image", "image": "file://data/1.jpg"}
    examples = [
      {
          "input": {"type": "image", "image": "file://data/0.jpg"},
          "output": '{"store": "WALMART"}'
      }
    ]
    """

    # prepare the user message content
    messages = [{"role": "user", "content": document}]
    # or in case of image: [{"role": "user", "content": [document]}]

    text = self.processor.tokenizer.apply_chat_template(
      messages,
      template=template,
      examples=examples,
      tokenize=False,
      add_generation_prompt=True,
    )

    # print("Formatted text input:")
    # print(text)

    # image_inputs = self._process_all_vision_info(messages)
    image_inputs = process_vision_info(messages)[0]
    # print("Image inputs:")
    # print(image_inputs)

    inputs = self.processor(
      text=[text],
      images=image_inputs,
      padding=True,
      return_tensors="pt",
    ).to(self.device)

    # we choose greedy sampling here, which works well for
    # most information extraction tasks
    generation_config = {
      "do_sample": False,
      "num_beams": 1,
      "max_new_tokens": output_cutoff,
    }

    # Inference: Generation of the output
    generated_ids = self.model.generate(**inputs, **generation_config)
    generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self.processor.batch_decode(
      generated_ids_trimmed,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False,
    )

    return output_text

  @staticmethod
  async def load_examples(input_file: Path) -> list[dict[str, str]]:
    # examples = [
    #   {
    #       "input": {"type": "image", "image": "file://data/0.jpg"},
    #       "output": '{"store": "WALMART"}'
    #   }
    # ]
    out = []
    contents = await asyncio.to_thread(input_file.read_text)
    example_parts = contents.split("\n<##NEW EXAMPLE##>\n")
    for ex_part in example_parts:
      parts = contents.split("\n<###EXPECTED OUTPUT###>\n")
      out.append({"input": parts[0], "output": parts[1]})
    return out
