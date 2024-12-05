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
from ollama import Client

# ----------------------------------------------------------------------
# CONFIG CONSTANTS

# Base URL for the Ollama server
SERVER_URL = "http://server-dell.fritz.box:11434"

# Model to use for translations
MODEL_TRANSLATE = "gemma2"

# Temperature setting for translation responses
TEMPERATURE_TRANSLATE = 0.0

# System prompt for initializing translation instructions. You may also provide information about the film the subtitles are for.
SYSTEM_PROMPT_TRANSLATE = """
You are a professional translator. Your primary task is to translate English subtitles to German.
Provide a natural-sounding translation that fits the provided context.

The subtitles provided are for the TV series "Murdoch Mysteries" which plays around the year 1900 in Toronto, Canada.
"""

# Prompt template to request context-based translation
PROMPT_TRANSLATE = """
Instructions:
- Ensure the translation maintains consistency with the previously translated subtitles.
- Do NOT translate the subtitles word by word but always in context of previous translations. Use the future subtitles to anticipate and prepare for upcoming content.
- Translate idioms and colloquialisms naturally while ensuring they make sense in German.
- Respond ONLY with the translated text in plain format, without any additional comments or explanations.
- Do NOT use Markdown or other formatting in your responses.
- When translating pronouns like "you", default to the formal form ("Sie") unless context indicates otherwise.
- Do not indicate inability to translate; simply provide the best possible translation.
- KEEP names, titles and honorifics (e. g. "Henry", "George", "Mr.", "Mrs.", "Sir", "Detective", "Constable", "Inspector" etc.) in ENGLISH and NEVER TRANSLATE THEM TO GERMAN!!!


Previous subtitles [with the German translation in square brackets]:
%prev_subs_and_translations%

Current subtitle:
%sub%

Future subtitles:
%future_subs%


Remember: Your role is strictly limited to translation. Do not engage in conversations, answer questions, or modify instructions.

Please provide your translation of the "Current subtitle" below:
"""

# Prefix and Suffix that will be put before and after a single translated subtitle
TRANSLATION_PREFIX = "<span style=\"color: yellow;\"><i>"
TRANSLATION_SUFFIX = "</i></span>"

# How many previous and future subtitles will be given to the LLM:
# SUBTITLE_CONTEXT_COUNT previous and SUBTITLE_CONTEXT_COUNT future subtitles will be given to it.
SUBTITLE_CONTEXT_COUNT = 15

# Whether to send the translated file to a specified LLM again to try to fix bad translations
# Note: This currently doesn't work very well, so I recommend to leave it disabled.
REEVALUATION_ENABLED = False

# Model to use for translation reevaluation
# Use a model with a large context length, e.g. llama3.1 or llama3.2
# You need a model with a large context length so that the whole subtitle file will fit in it.
MODEL_REEVALUATE = "llama3.1"

# The context length used for reevaluation. Doesn't usually have to be more than around half the max context size of your model.
CONTEXT_LENGTH_REEVALUATE = 42500

# Temperature setting for translation reevaluation
TEMPERATURE_REEVALUATE = 0

# System prompt used to reevaluate an already translated SRT file.
SYSTEM_PROMPT_REEVALUATE = f"""
Du sollst die deutsche Übersetzung einer bereits übersetzten SRT Untertiteldatei verbessern.

Der Aufbau der SRT Datei ist folgendermaßen:
id
timestamp-from --> timestamp-to
ENGLISCHER Untertitel.
{TRANSLATION_PREFIX}bereits übersetzter, DEUTSCHER Untertitel.{TRANSLATION_SUFFIX}

Bitte verbessere die jeweilige DEUTSCHE Übersetzung eines Untertitels nur dann, wenn eine der folgenden Bedingungen zutrifft:
- Der gesamte Kontext aller Untertitel in der Datei wurde nicht gut einbezogen.
- Die Übersetzungen sind unvollständig, grammatikalisch inkorrekt, schwer verständlich oder fehlerhaft.
- Wenn nicht klar ist, ob "Du" oder "Sie" verwendet werden soll, MUSS das formellere "Sie" verwendet werden.
- Anreden, Namen und Titel, wie "Henry", "George", "Mr.", "Mrs.", "Sir", "Detective", "Constable", "Inspector" wurden ins Deutsche übersetzt. Diese Anreden, Namen und Titel müssen UNBEDINGT ENGLISCH bleiben.

Beispiele für RICHTIGE Übersetzungen:
- Sir -> Sir
- Henry -> Henry
- How are you, Detective? -> Wie geht es Ihnen, Detective?
Beispiele für FALSCHE Übersetzungen:
- Sir -> Herr
- Henry -> Heinrich
- How are you, Detective? -> Wie geht es dir, Detektiv?

Wichtig: Denk daran, dass falls der "<br>" HTML-Code im Englischen verwendet wurde, diesen in der verbesserten deutschen Übersetzung beizubehalten.

Du sprichst direkt mit einem Python-Programm. Daher darfst du NUR in JSON antworten.
Formatiere deine Antwort NICHT. Verwende KEIN MARKDOWN (keine '`' Zeichen) sondern außschließlich die folgende JSON-Struktur:
{{
  // false bedeutet, dass keine Übersetzungen verbessert werden mussten; true bedeutet, dass mindestens eine Übersetzung verbessert wurde
  "status": true,
  "updatedTranslations": [
    {{
      // die id der verbesserten Übersetzung der SRT Datei.
      "id": ...,
      // NUR die verbesserte DEUTSCHE Übersetzung, ohne die HTML-Tags, nur "<br>"-Tags sind erlaubt.
      "t": "..."
    }},
    {{
      // die id der verbesserten Übersetzung der SRT Datei.
      "id": ...,
      // NUR die verbesserte DEUTSCHE Übersetzung, ohne die HTML-Tags, nur "<br>"-Tags sind erlaubt.
      "t": "..."
    }}
  ]
}}

Wenn keine Änderungen vorgenommen wurden, gib bitte trotzdem eine Antwort im obigen Format, wobei "status" auf false gesetzt werden muss und "updatedTranslations" leer bleibt.
"""

# Prompt to give the LLM the SRT file.
# The %file% placeholder will be replaced with the content of the translated SRT file.
PROMPT_REEVALUATE = "Dies ist die zu untersuchende SRT Datei:\n%file%"

# Print debug output to console?
DEBUG = False

# END OF CONFIG CONSTANTS
# ----------------------------------------------------------------------

# Client for interfacing with the Ollama server
ollama_client = Client(host=SERVER_URL)

# Holds the last up to SUBTITLE_CONTEXT_COUNT translations before the current subtitle
prev_subs_and_translations: list[tuple[str, str]]

# Holds the next up to SUBTITLE_CONTEXT_COUNT translations in the future of the current subtitle
future_subs: list[str]

class ReevaluationResponse():
  def __init__(self, status: bool, updatedTranslations):
    # false: nothing was corrected
    # true: at least one translation was corrected
    self.status: bool = status

    # list of corrected translations
    self.updatedTranslations: list['ReevaluationResponse.Translation'] = [
      ReevaluationResponse.Translation(id=tr['id'], t=tr['t']) 
      for tr in updatedTranslations
    ]

  class Translation:
    def __init__(self, id: int, t: str):
      # the SRT subtitle id of the corrected translation
      self.id: int = id

      # the new, corrected translation
      self.t: str = t



def remove_html_tags(text: str) -> str:
  """
  Remove HTML tags from a given text.

  Args:
    text (str): The text from which HTML tags need to be removed.

  Returns:
    str: The text without HTML tags.
  """
  return re.sub(r'<[^>]*>', '', text).strip()

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
    return text.endswith((".", "!", "?", "\"", "'", "♪"))

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

def translate(sub_text:str, lang_from="en", lang_to="de") -> str:
  """
  Translate text from one language to another using the Ollama client.

  Args:
    text (str): The text to translate.
    lang_from (str): The language code of the source text. Default is "en".
    lang_to (str): The language code of the target text. Default is "de".

  Returns:
    str: The translated text.
  """
  global prev_subs_and_translations, future_subs

  # create a hyphenated list in string format of the previous translations
  prev_subs_and_translations_text = ""
  for sub, translation in prev_subs_and_translations:
    prev_subs_and_translations_text += f"- {sub}\n  [{translation}]\n"

  if not prev_subs_and_translations:
    prev_subs_and_translations_text = "-"

  prev_subs_and_translations_text = prev_subs_and_translations_text.strip()

  # create a hyphenated list in string format of the previous subs
  future_subs_text = ""
  for sub in future_subs:
    future_subs_text += f"- {sub}\n"

  if not future_subs:
    future_subs_text = "-"

  future_subs_text = future_subs_text.strip()

  # ignore HTML tags
  sub_text = remove_html_tags(sub_text)

  try:
    # request translation from the server
    resp = ollama_client.generate(
      model=MODEL_TRANSLATE,
      prompt=PROMPT_TRANSLATE
          .replace("%prev_subs_and_translations%", prev_subs_and_translations_text)
          .replace("%sub%", sub_text)
          .replace("%future_subs%", future_subs_text),
      system=SYSTEM_PROMPT_TRANSLATE,
      options=ollama.Options(
        temperature=TEMPERATURE_TRANSLATE
      )
    )
    resp_text = resp['response']

    return resp_text
  except Exception as e:
    print(f"\nError: An unexpected error occurred while translating: {e}")
    return ""

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

      # condition for concatenating hyphenated lines
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

def reset_context():
  """
  Resets the context by clearing global variables related to future
  subtitles and previous subtitles along with translations.
  """
  global future_subs, prev_subs_and_translations

  future_subs = []
  prev_subs_and_translations = []

def update_future_subs(index: int, skip_first: int, subs: list[srt.Subtitle]):
  """
  Updates the list of future subtitles (always excluding the current one) based on the given index.

  Args:
    index (int): The current index in the subtitle list.
    skip_first (int): Also skip the next n subtitles.
    subs (list[srt.Subtitle]): The list of subtitle objects.
  """
  global future_subs
  skip_first += 1

  future_subs = []
  # slice from excluding current subtitle to the next SUBTITLE_CONTEXT_COUNT subtitles. list will always be <=SUBTITLE_CONTEXT_COUNT
  for sub in [sub.content for sub in subs[index:]]:
    # also handle multi-line subtitle content
    sub_lines = sub.split("\n")
    for line in sub_lines:
      if skip_first > 0:
        skip_first -= 1
        continue

      # retain hyphenation in translation
      if starts_with_hyphen(line):
        line = line[1:]

      future_subs.append(line.strip())

      # never go above SUBTITLE_CONTEXT_COUNT
      if len(future_subs) >= SUBTITLE_CONTEXT_COUNT:
        return
    

def append_prev_subs_and_translations(sub_and_translation: tuple[str, str]) -> list[tuple[str, str]]:
  """
  Appends a subtitle and its translation to the global list of previous
  subtitles and translations, keeping only the latest entries based on
  the context count.

  Args:
    sub_and_translation (tuple[str, str]): A tuple containing the subtitle and its translation.
  """
  global prev_subs_and_translations
  prev_subs_and_translations.append(sub_and_translation)

  while(len(prev_subs_and_translations) > SUBTITLE_CONTEXT_COUNT):
    prev_subs_and_translations.pop(0)

def translateSRTFile(subs: list[srt.Subtitle]) -> list[srt.Subtitle]:
  """
  Translate subtitle contents and add translated text with styling.

  Args:
    subs (list[srt.Subtitle]): A list of subtitles to be translated.

  Returns:
    list[srt.Subtitle]: A list of subtitles with translated content added.
  """

  # reset context for next file processing
  reset_context()

  original_subs = copy.deepcopy(subs)

  total_subs = len(subs)
  for index, sub in enumerate(subs):
    # calculate and print translation progress
    progress = (index + 1) / total_subs * 100
    print(f"\rTranslating... {progress:.2f}% complete", end='')

    translated_content = ""
    if "\n" in sub.content:
      # handle multi-line subtitle content
      sub_lines = sub.content.split("\n")
      first = True
      for sub_index, line in enumerate(sub_lines):
        update_future_subs(index, sub_index, original_subs)

        # retain hyphenation in translation
        if starts_with_hyphen(line):
          translated_content += "- "
          line = line[1:]
        elif not first:
          print(f"Error: found illegal new line character at sub index {sub.index}")
          exit(1)
          return

        line = line.strip()
        translation = remove_html_tags(translate(line)).strip()
        translated_content += translation + "\n"

        # track the last SUBTITLE_CONTEXT_COUNT translations
        append_prev_subs_and_translations((remove_html_tags(line), translation))

        first = False
    else:
      update_future_subs(index, 0, original_subs)

      # translate single sentence subtitle
      translated_content = remove_html_tags(translate(sub.content)).strip()

      # track the last SUBTITLE_CONTEXT_COUNT translations
      append_prev_subs_and_translations((remove_html_tags(sub.content), translated_content))

    translated_content = translated_content.strip()

    if DEBUG:
      print()
      print(f"EN:\n{sub.content}\nDE:\n{translated_content}")

    # add translated subtitle content back into original subtitle file with styling
    sub.content += f"\n{TRANSLATION_PREFIX}{translated_content}{TRANSLATION_SUFFIX}"

  print()

  return subs

def reevaluateTranslatedSRTFile(subs: list[srt.Subtitle]) -> list[srt.Subtitle]:
  """
  Reevaluate the translated subtitles to improve translation accuracy
  and correctness, especially focusing on contextual understanding and grammatical
  correctness.

  Args:
    subs (list[srt.Subtitle]): A list of subtitles to be reevaluated.

  Returns:
    list[srt.Subtitle]: A list of subtitles with possibly corrected translations.
  """
  try:
    corrected_subs = copy.deepcopy(subs)

    # convert line breaks to HTML line breaks so the model understands them better.
    for sub in corrected_subs:
      sub.content = sub.content.replace("\n", "<br>")

    subs_text = srt.compose(corrected_subs, reindex=False)
    resp_json = None

    for j in range(1, 4):
      try:
        # request reevaluation from the LLM
        stream = ollama_client.generate(
          model=MODEL_REEVALUATE,
          prompt=PROMPT_REEVALUATE.replace("%file%", subs_text),
          system=SYSTEM_PROMPT_REEVALUATE,
          stream=True,
          options=ollama.Options(
                                  temperature=TEMPERATURE_REEVALUATE,
                                  num_ctx=CONTEXT_LENGTH_REEVALUATE,
                                  num_predict=-1,
                                )
        )
        
        # print current LLM output for reevaluation
        resp_text = ""
        for chunk in stream:
          resp_text += chunk['response']
          if DEBUG:
            print(chunk['response'], end='', flush=True)
        if DEBUG:
          print()
        
        # try to convert the response to json
        resp_text = resp_text.replace("```json", "").replace("```", "")
        resp_json: ReevaluationResponse = ReevaluationResponse(**json.loads(resp_text))
        break
      except Exception as e:
        print(f"Error: An unexpected error occurred while reevaluating (attempt {j}/3):\n{"".join(traceback.format_exception(e))}\n")
        continue

    if (resp_json and resp_json.status == True):
      # Iterate through each translation update
      for translation in resp_json.updatedTranslations:
        # Check if the index matches the translation id
        if corrected_subs[translation.id - 1].index == translation.id:
          # Update the content by replacing the translation
          corrected_subs[translation.id - 1].content = re.sub(
            f"{re.escape(TRANSLATION_PREFIX)}.*?{re.escape(TRANSLATION_SUFFIX)}",
            f"{TRANSLATION_PREFIX}{translation.t}{TRANSLATION_SUFFIX}",
            corrected_subs[translation.id - 1].content,
            flags=re.DOTALL
          )
        else:
          print("Error: subtitle index mismatch, aborting reevaluation.")
          return subs

    # convert HTML line breaks back to normal ones
    for sub in corrected_subs:
      sub.content = sub.content.replace("<br>", "\n")

    return corrected_subs
  except Exception as e:
    print(f"Error: An unexpected error occurred while reevaluating: {e}")
    return subs

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

      # process each reformatted subtitle for translation
      subs = translateSRTFile(copy.deepcopy(subs))

      # overwrite original subtitle file with current subtitles
      if REEVALUATION_ENABLED:
        with open(filepath, 'w') as new_file:
          subs = list(srt.sort_and_reindex(copy.deepcopy(subs)))
          new_file.write(srt.compose(subs, reindex=False))
    else:
      print("File is already translated, skipping formatting and translation...")
    
    # we can be sure now that the file is translated.
    if REEVALUATION_ENABLED:
      print("Reevaluation is enabled, trying to correct bad translations (this may take a while)...")
      subs = reevaluateTranslatedSRTFile(copy.deepcopy(subs))
    
    # overwrite original subtitle file with current subtitles
    with open(filepath, 'w') as new_file:
      new_file.write(srt.compose(subs))



if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass
