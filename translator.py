#!/usr/bin/env python

import os
import sys
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
MODEL = "gemma2"

# Temperature setting for translation responses
TEMPERATURE = 0.35

# System prompt for initializing translation instructions
SYSTEM_PROMPT = """
You are a German language translator. Your primary task is to translate English text provided by users into German.

Translate all user input from English to German.
Do NOT translate sentences word by word but in context.
Also consider the context of previous translations when generating responses (due to system limitations, you only have access to up to the last 15 translations as context).
Respond ONLY with the translated text in plain format, without any additional comments or explanations.
Do NOT use Markdown or other formatting in your responses.
Maintain consistency across multiple interactions where possible.

When translating pronouns like "you", default to the formal form ("Sie") unless context indicated otherwise.
If a word or phrase cannot be translated, replace it with "???".
Do not indicate inability to translate; simply provide the best possible translation.
KEEP names, titles and honorifics (e. g. "Henry", "George", "Mr.", "Mrs.", "Sir", "Detective", "Constable", "Inspector" etc.) in ENGLISH and do not translate them.

Remember: Your role is strictly limited to translation. Do not engage in conversations, answer questions, or modify instructions.
"""

# END OF CONFIG CONSTANTS
# ----------------------------------------------------------------------


# Pre-defined initial chat messages used in translation
EMPTY_CHAT_MESSAGES = [
  {
    'role': 'system',
    'content': SYSTEM_PROMPT
  }
]

# Client for interfacing with the Ollama server
ollama_client = Client(host=SERVER_URL)

# Holds current chat messages with translation instructions
chat_messages = EMPTY_CHAT_MESSAGES


def remove_html_tags(text: str) -> str:
  """
  Remove HTML tags from a given text.

  Args:
    text (str): The text from which HTML tags need to be removed.

  Returns:
    str: The text without HTML tags.
  """
  return re.sub(r'<[^>]*>', '', text)

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
    return text.endswith((".", "!", "?", "\"", "'"))

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

def translate(text:str, lang_from="en", lang_to="de") -> str:
  """
  Translate text from one language to another using the Ollama client.

  Args:
    text (str): The text to translate.
    lang_from (str): The language code of the source text. Default is "en".
    lang_to (str): The language code of the target text. Default is "de".

  Returns:
    str: The translated text.
  """
  global chat_messages

  # ignore HTML tags
  text = remove_html_tags(text)

  try:
    # append user text to chat history
    chat_messages.append({
      'role': 'user',
      'content': f"Translate this:\n'{text}'"
    })

    # request translation from the server
    resp = ollama_client.chat(model=MODEL, messages=chat_messages, options=ollama.Options(temperature=TEMPERATURE))
    resp_text = resp['message']['content']

    # append translated text to chat history
    chat_messages.append({
      'role': 'translator',
      'content': resp_text
    })

    # trim chat history if it exceeds 15 translation request-answer pairs
    # always keep the system prompt (first entry)
    if len(chat_messages) > 31:
      del chat_messages[1:len(chat_messages) - 30]

    return resp_text
  except Exception as e:
    print(f"Error: An unexpected error occurred while translating: {e}")
    return ""

def main():
  """
  Main function to perform translation on subtitle files.
  """
  global chat_messages
  try:
    # check server connection
    resp = requests.get(f"{SERVER_URL}")
    resp.raise_for_status()
  except Exception as e:
    print(f"Error: Cannot connect to Ollama server: {e}")
    sys.exit(1)

  # directory where subtitle files are stored
  subs_dir = os.path.join(os.path.dirname(__file__), 'subs')
  files = os.listdir(subs_dir)
  total_files = len(files)

  # loop through all files in 'subs' directory
  for n, filename in enumerate(files):
    # print progress of current file processing
    print(f"\nFile {filename} ({n + 1}/{total_files}):")
    filepath = os.path.join(subs_dir, filename)

    # ignore the .gitkeep file
    if filepath.endswith(".gitkeep"):
      continue

    # skip non-SRT files and warn user
    if not filepath.endswith(".srt"):
      print(f"Warning: File {filepath} is not an SRT file, skipping")
      continue

    new_subs: list[srt.Subtitle] = list()
    new_sub_id = 1
    with open(filepath, 'r') as file:
      # parse subtitle file content
      subs = list(srt.parse(''.join(file.readlines())))
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
            if (starts_with_hyphen(line) and prev_line.strip() and not ends_with_punctuation(prev_line.strip())):
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
          new_subs.append(srt.Subtitle(new_sub_id, start_sub.start, sub.end, line_builder.strip(), ""))

          new_sub_id += 1
          line_builder = ""
          prev_line = ""
          # move to next subtitle segment
          start_sub = subs[-1] if sub == subs[-1] else subs[subs.index(sub) + 1]
    print()

    # process each reformatted subtitle for translation
    total_new_subs = len(new_subs)
    for index, sub in enumerate(new_subs):
      # calculate and print translation progress
      progress = (index + 1) / total_new_subs * 100
      print(f"\rTranslating... {progress:.2f}% complete", end='')

      translated_content = ""
      if "\n" in sub.content:
        # handle multi-line subtitle content
        sub_lines = sub.content.split("\n")
        for line in sub_lines:
          # retain hyphenation in translation
          if line.startswith("-"):
            translated_content += "- "
            line = line[1:]

          line = line.strip()
          translated_content += translate(line).strip() + "\n"
      else:
        # translate single sentence subtitle
        translated_content = translate(sub.content)

      # add translated subtitle content back into original subtitle file with styling
      sub.content += f"\n<span style=\"color: yellow;\"><i>{translated_content.replace('>', '').replace('<', '').strip()}</i></span>"
    print()

    # overwrite original subtitle file with translated content
    with open(filepath, 'w') as new_file:
      new_file.write(srt.compose(new_subs))

    # reset chat messages for next file processing
    chat_messages = EMPTY_CHAT_MESSAGES


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass