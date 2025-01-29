#!/usr/bin/env python

import copy
import json
import os
import sys
import traceback
import ollama
import requests
import srt
import re
import ast
from ollama import Client

# ----------------------------------------------------------------------
# CONFIG CONSTANTS

# Base URL for the Ollama server
SERVER_URL = "http://server-dell.fritz.box:11434"

# Model to use for translations. Recommended: deepseek-r1:14b (slow, better quality), gemma2:9b-instruct-q4_K_M (fast, medium quality)
MODEL_TRANSLATE = "gemma2:9b-instruct-q4_K_M"

# If the above model fails to output a valid JSON array, fall back to this model.
# Should be smarter than above model. Set to same model to disable.
MODEL_TRANSLATE_FALLBACK = "deepseek-r1:14b"

# Temperature setting for translation responses
TEMPERATURE_TRANSLATE = 0.0

# Temperature setting for translation responses
TEMPERATURE_TRANSLATE_FALLBACK = 0.6

# System prompt for initializing translation instructions.
SYSTEM_PROMPT_TRANSLATE = ""

# Prompt template to request context-based translation
PROMPT_TRANSLATE = """
Hello, I would like you to professionally translate subtitles from English to German.
The subtitles provided are for the TV crime-series "Murdoch Mysteries" which plays around the year 1900 in Toronto, Canada.
When translating pronouns like "you", please default to the formal form ("Sie") unless context indicates otherwise.
NEVER alter names, titles, exclamations and honorifics like "Henry", "George", "Mr.", "Mrs.", "Sir", "Detective", "Constable", "Inspector", "Oh", "Aah" and the like. KEEP THEM IN ENGLISH!
Make sure that you translate idioms and colloquial expressions in such a way that they make sense in German.
Please provide a natural-sounding German translation that fits the provided context (previous and upcoming subtitles) while ensuring the translation maintains consistency with the previously translated subtitles.
Do NOT translate the subtitles word for word. Use the provided previous subtitles and translations as well as the upcoming translations to understand what is happening in the series.
Please do not indicate inability to translate; simply provide the best possible translation.


Context 1 (previous subtitles and translations):
%prev_subs_and_translations%


Context 2 (upcoming subtitles):
%future_subs%


The subtitles I want you to translate:
%subs%


Respond with the translated subtitles in a JSON array, without any additional comments or explanations.
Only use plaintext inside the translations and do NOT use Markdown or other formatting in your response.

Example of the JSON structure:
["Translation of Subtitle 1", "Translation of Subtitle 2", "Translation of Subtitle 3", ...]

Remember: Your role is strictly limited to translation. Do not engage in conversations, answer questions, or modify instructions.
Please provide your translations below. Thank you!
"""

# Prefix and Suffix that will be put before and after a single translated subtitle
# Prefix must be provided and be unique string in the subtitle. Use e. g. HTML tags like the default values.
TRANSLATION_PREFIX = "<span style=\"color: yellow;\"><i>"
# Suffix is optional
TRANSLATION_SUFFIX = "</i></span>"

# How many previous and future subtitles will be given to the LLM:
# SUBTITLE_CONTEXT_COUNT previous and SUBTITLE_CONTEXT_COUNT future subtitles will be given to it.
SUBTITLE_CONTEXT_COUNT = 10

# How many subtitles will be given to the LLM at once.
TRANSLATION_BATCH_LENGTH = 10

# Print debug output to console?
DEBUG = True

# END OF CONFIG CONSTANTS
# ----------------------------------------------------------------------

# Client for interfacing with the Ollama server
ollama_client = Client(host=SERVER_URL)

# Holds the last up to SUBTITLE_CONTEXT_COUNT translations before the current subtitle
prev_subs_and_translations: list[tuple[str, str]]

# Holds the next up to SUBTITLE_CONTEXT_COUNT translations in the future of the current subtitle
future_subs: list[str]


def remove_html_tags(text: str) -> str:
  """
  Remove HTML tags from a given text.

  Args:
    text (str): The text from which HTML tags need to be removed.

  Returns:
    str: The text without HTML tags.
  """
  return re.sub(r'<think>(.|\n)*</think>', '', text).strip()

def remove_thinking(text: str) -> str:
  """
  Remove XML thinking from a given text (LLM response).

  Args:
    text (str): The text from which XML thinking needs to be removed.

  Returns:
    str: The text without the thinking tags and content.
  """
  return re.sub(r'<think>(.|\n)*</think>', '', text).strip()

def ends_with_punctuation(text: str) -> bool:
  """
  Check if the given text ends with a punctuation mark.

  Args:
    text (str): The text to check.

  Returns:
    bool: True if the text ends with a punctuation mark, False otherwise.
  """
  # ignore HTML tags
  text = remove_html_tags(text)
  return text.endswith((".", "!", "?", "\"", "'", "♪", "]"))

def starts_with_hyphen(text: str) -> bool:
    """
    Check if the given text starts with a hyphen.

    Args:
      text (str): The text to check.

    Returns:
      bool: True if the text starts with a hyphen, False otherwise.
    """
    # ignore HTML tags
    text = remove_html_tags(text)
    return text.startswith("-")

def reset_context():
  """
  Resets the context by clearing global variables related to future
  subtitles and previous subtitles along with translations.
  """
  global future_subs, prev_subs_and_translations

  future_subs = []
  prev_subs_and_translations = []

def update_future_subs(index: int, subs: list[srt.Subtitle]):
  """
  Updates the list of future subtitles (always excluding the current one) based on the given index.

  Args:
    index (int): The current index in the subtitle list.
    subs (list[srt.Subtitle]): The list of subtitle objects.
  """
  global future_subs

  future_subs = []
  # slice from excluding current subtitle to the next SUBTITLE_CONTEXT_COUNT subtitles. list will always be <=SUBTITLE_CONTEXT_COUNT
  for sub in [sub.content for sub in subs[index:]]:
    future_subs.append(sub.strip())

    # never go above SUBTITLE_CONTEXT_COUNT
    if len(future_subs) >= SUBTITLE_CONTEXT_COUNT:
      return
    

def update_previous_subs_and_translations(index: int, subs: list[srt.Subtitle]):
  """
  Updates the list of previous subtitles and translations based on the given index.

  Args:
      index (int): The current index in the subtitle list.
      subs (list[srt.Subtitle]): The list of subtitle objects.
  """
  global prev_subs_and_translations

  prev_subs_and_translations = []

  start_index = max(0, index - SUBTITLE_CONTEXT_COUNT)
  for sub in subs[start_index:index]:
    content_parts = sub.content.split(TRANSLATION_PREFIX, 1)
    
    if len(content_parts) == 2:
      sub_content, sub_translation = content_parts
      
      if sub_translation.endswith(TRANSLATION_SUFFIX):
        # make sure to only remove suffix at end
        sub_translation = sub_translation[:-len(TRANSLATION_SUFFIX)]
      
      prev_subs_and_translations.append(
        (sub_content.strip(), sub_translation.strip())
      )
    
    if len(prev_subs_and_translations) >= SUBTITLE_CONTEXT_COUNT:
      break

def is_valid_list(obj) -> bool:
  """
  Check if an object is a list of strings or a list of lists of strings.

  Args:
    obj: The object to be checked. Can be any type.

  Returns:
    bool: True if the object is a valid list of strings or nested lists of strings, False otherwise.
"""
  if not isinstance(obj, (list, tuple)):
    return False
  
  for item in obj:
    if not isinstance(item, (str, list, tuple)):
      return False
    
    if isinstance(item, (list, tuple)):
      for sub_item in item:
        if not isinstance(sub_item, str):
          return False
  
  return True


def prompt_model(prompt:str, required_response_length:int, model:str, temp:float):
  """
  Request a translation from the server using the Ollama client, ensuring
  that the response matches the required length.

  Args:
    prompt (str): The prompt or query to be sent to the server for translation.
    required_response_length (int): The expected number of translations to be returned.
    model (str): The model to be used for generating translations.
    temp (float): The temperature setting for the generation process.

  Raises:
    Exception: If the number of translations in the response does not match the required length.

  Returns:
    list[str]: A list of translations received from the server.
  """
  # request translation from the server
  stream = ollama_client.generate(
    model=model,
    prompt=prompt,
    system=SYSTEM_PROMPT_TRANSLATE,
    stream=True,
    options=ollama.Options(
      temperature=temp
    )
  )

  if DEBUG:
    print("---------------- RESPONSE ----------------")

  resp_text = ""
  for chunk in stream:
    resp_text += chunk['response']
    if DEBUG:
      print(chunk['response'], end='', flush=True)

  if DEBUG:
    print("\n-------------- END RESPONSE --------------")

  resp_list = ast.literal_eval(remove_thinking(resp_text))

  if len(resp_list) != required_response_length:
    raise Exception(f"LLM did not return correct amount of translations. Required: {required_response_length}. Got: {len(resp_list)}.")

  if not is_valid_list(resp_list):
    raise Exception("LLM did not return a valid list.")

  return resp_list

def translate_batch(subs_batch:list[srt.Subtitle]):
  """
  Translate text from one language to another using the Ollama client.

  Args:
    text (str): The text to translate.

  Returns:
    str: The translated text.
  """
  global prev_subs_and_translations, future_subs

  # create a list in string format of numbered previous subs and translations
  prev_subs_and_translations_text = ""
  for sub, translation in prev_subs_and_translations:
    prev_subs_and_translations_text += f"- '{remove_html_tags(sub).replace("\n", "\\n")}'\n  Translation: '{translation.replace("\n", "\\n")}'\n"

  if not prev_subs_and_translations:
    prev_subs_and_translations_text = "No previous subtitles available."

  prev_subs_and_translations_text = prev_subs_and_translations_text.strip()

  # create a list in string format of numbered subs to translate
  # subtitle id
  id = 1
  subs_text = ""
  for sub in subs_batch:
    subs_text += f"- Sutitle {id}: '{remove_html_tags(sub.content).replace("\n", "\\n")}'\n"
    id += 1

  subs_text = subs_text.strip()

  # create a list in string format of numbered upcoming subs
  future_subs_text = ""
  for sub in future_subs:
    future_subs_text += f"- '{remove_html_tags(sub).replace("\n", "\\n")}'\n"

  if not future_subs:
    future_subs_text = "No future subtitles available."

  future_subs_text = future_subs_text.strip()

  prompt = (PROMPT_TRANSLATE.replace("%prev_subs_and_translations%", prev_subs_and_translations_text)
                            .replace("%subs%", subs_text)
                            .replace("%future_subs%", future_subs_text))

  if DEBUG:
    print()
    print("---------------- PROMPT ----------------")
    print(prompt)
    print("-------------- END PROMPT --------------")

  # retry default model 5 times
  for j in range(5):
    try:
      return prompt_model(prompt, len(subs_batch), MODEL_TRANSLATE, TEMPERATURE_TRANSLATE)
    except Exception as e:
      print(f"\nError: An error occurred while translating with model '{MODEL_TRANSLATE}' (Attempt {j + 1}/5): {e}")

  print("Retrying with fallback model...")

  # retry fallback model 5 times
  for j in range(5):
    try:
      return prompt_model(prompt, len(subs_batch), MODEL_TRANSLATE_FALLBACK, TEMPERATURE_TRANSLATE_FALLBACK)
    except Exception as e:
      print(f"\nError: An error occurred while translating with fallback model '{MODEL_TRANSLATE_FALLBACK}' (Attempt {j + 1}/5): {e}")

  raise Exception("Max retry amount reached.")

def translateSRTFile(subs: list[srt.Subtitle], filepath: str) -> list[srt.Subtitle]:
  """
  Translate subtitle contents and add translated text with styling.

  Args:
    subs (list[srt.Subtitle]): A list of subtitles to be translated.
    filepath (str): The path to the subtitle file.

  Returns:
    list[srt.Subtitle]: A list of subtitles with translated content added.
  """

  # reset context for next file processing
  reset_context()

  total_subs = len(subs)
  for startIndex in range(0, len(subs), TRANSLATION_BATCH_LENGTH):
    # skip already translated subs
    if (TRANSLATION_PREFIX in subs[startIndex].content or TRANSLATION_SUFFIX in subs[startIndex].content):
      continue

    # calculate and print translation progress
    progress = (startIndex) / total_subs * 100
    print(f"\rTranslating... {progress:.2f}% complete", end='')

    subs_batch = subs[startIndex:startIndex+TRANSLATION_BATCH_LENGTH]

    update_previous_subs_and_translations(startIndex, subs)

    update_future_subs(startIndex + TRANSLATION_BATCH_LENGTH, subs)

    # translate subtitle batch
    translations = translate_batch(subs_batch)

    for sub, translated_content in zip(subs_batch, translations):
      if isinstance(translated_content, (list, tuple)):
        translated_content = "\n".join(translated_content)
      translated_content = translated_content.strip().replace("\\n", "\n")

      translated_content = translated_content.strip()

      # add translated subtitle content back into original subtitle file with styling
      sub.content += f"\n{TRANSLATION_PREFIX}{translated_content}{TRANSLATION_SUFFIX}"
    
    with open(filepath, 'w') as new_file:
        new_file.write(srt.compose(subs))

  print("Translating... 100.00% complete")

  return subs

def reformatSRTFile(subs: list[srt.Subtitle]) -> list[srt.Subtitle]:
  """
  Reformat subtitles by merging lines based on punctuation and hyphenation rules.

  Args:
    subs (list[srt.Subtitle]): A list of subtitles to be reformatted.

  Returns:
    list[srt.Subtitle]: A list of reformatted subtitles.
  """
  formatted_subs: list[srt.Subtitle] = list()
  formatted_sub_id = 1
  total_subs = len(subs)

  prev_line = ""
  line_builder = ""
  start_sub = subs[0]
  for index, sub in enumerate(subs):
    # calculate and print reformatting progress
    progress = (index + 1) / total_subs * 100
    print(f"\rReformatting... {progress:.2f}% complete", end='')

    sub_lines = sub.content.split("\n")

    for line in sub_lines:
      # remove leading and trailing whitespace
      line = line.strip()

      # condition for concatenating hyphenated lines and removing new lines otherwise
      if (starts_with_hyphen(line) and prev_line.strip() and not ends_with_punctuation(prev_line)):
        line = line[1:].strip()
        line_builder += " "
      elif (starts_with_hyphen(line)):
        line_builder += "\n"
      else:
        line_builder += " "

      line_builder += line
      prev_line = line

    # condition to create new subtitle entry
    if (ends_with_punctuation(prev_line)):
      formatted_subs.append(srt.Subtitle(formatted_sub_id, start_sub.start, sub.end, line_builder.strip(), ""))

      formatted_sub_id += 1
      line_builder = ""
      prev_line = ""
      # move to next subtitle segment
      start_sub = subs[-1] if sub == subs[-1] else subs[subs.index(sub) + 1]
  print()

  return formatted_subs

def main():
  """
  Main function to perform translation on subtitle files.
  """
  try:
    # check server connection
    resp = requests.get(f"{SERVER_URL}")
    resp.raise_for_status()
  except Exception as e:
    print(f"Error: Cannot connect to Ollama server: {e}")
    sys.exit(1)

  # directory where subtitle files are stored
  subs_dir = os.path.join(os.path.dirname(__file__), 'subs')

  # remove files ending in .gitkeep
  files = sorted([f for f in os.listdir(subs_dir) if not f.endswith('.gitkeep')])
  total_files = len(files)

  # loop through all files in 'subs' directory
  for n, filename in enumerate(files):
    filepath = os.path.join(subs_dir, filename)

    # print progress of current file processing
    print(f"\nFile {filename} ({n + 1}/{total_files}):")

    # skip non-SRT files and warn user
    if not filepath.endswith(".srt"):
      print(f"Warning: File {filepath} is not an SRT file, skipping")
      continue

    subs: list[srt.Subtitle]
    with open(filepath, 'r') as file:
      # parse subtitle file content
      subs = list(srt.parse(''.join(file.readlines())))

    # check if subtitle file is already translated
    if (not TRANSLATION_PREFIX in subs[0].content and not TRANSLATION_SUFFIX in subs[0].content):
      subs = reformatSRTFile(copy.deepcopy(subs))

      # overwrite original subtitle file with current subtitles
      with open(filepath, 'w') as new_file:
        new_file.write(srt.compose(subs))

    if (not TRANSLATION_PREFIX in subs[-1].content and not TRANSLATION_SUFFIX in subs[-1].content):
      # process each reformatted subtitle for translation
      subs = translateSRTFile(copy.deepcopy(subs), filepath)
    else:
      print("File is already translated, skipping formatting and translation...")
    
    # overwrite original subtitle file with current subtitles
    with open(filepath, 'w') as new_file:
      new_file.write(srt.compose(subs))



if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass
