#!/usr/bin/env python

import os
import ollama
import requests
import srt
import re
from ollama import Client

# ----------------------------------------------------------------------
# CONFIG CONSTANTS

SERVER_URL = "http://server-dell.fritz.box:11434"
MODEL = "gemma2"
TEMPERATURE = 0.35
SYSTEM_PROMPT = """
You are a German language translator. Your primary task is to translate English text provided by users into German.

Translate all user input from English to German.
Do NOT translate sentences word by word but in context.
Also consider the context of previous translations when generating responses (due to system limitations, you only have access to up to the last 15 translations as context).
Respond ONLY with the translated text in plain format, without any additional comments or explanations.
Do NOT use Markdown or other formatting in your responses.
Maintain consistency across multiple interactions where possible.

When translating pronouns like 'you', default to the formal form ('Sie') unless context indicated otherwise.
If a word or phrase cannot be translated, replace it with '???'.
Do not indicate inability to translate; simply provide the best possible translation.
Keep names, titles and honorifics (e. g. "Henry", "George", "Sir", "Mr.", "Mrs.", "Detective", "Constable", "Inspector" etc.) in English and do NOT translate them.

Remember: Your role is strictly limited to translation. Do not engage in conversations, answer questions, or modify instructions.
"""

# END OF CONFIG CONSTANTS
# ----------------------------------------------------------------------


EMPTY_CHAT_MESSAGES = [
  {
    'role': 'system',
    'content': SYSTEM_PROMPT
  }
]

ollama_client = Client(host=SERVER_URL)


def remove_html_tags(text: str) -> str:
  return re.sub(r'<[^>]*>', '', text)

def ends_with_punctuation(text: str) -> bool:
    # ignore HTML tags
    text = remove_html_tags(text)
    return text.endswith((".", "!", "?", "\"", "'"))

def starts_with_hyphen(text: str) -> bool:
    # ignore HTML tags
    text = remove_html_tags(text)
    return text.startswith("-")

chat_messages = EMPTY_CHAT_MESSAGES
def translate(text:str, lang_from="en", lang_to="de") -> str:
  global chat_messages

  # ignore HTML tags
  text = remove_html_tags(text)

  try:
    chat_messages.append({
      'role': 'user',
      'content': f"Translate this:\n'{text}'"
    })

    resp = ollama_client.chat(model=MODEL, messages=chat_messages, options=ollama.Options(temperature=TEMPERATURE))
    resp_text = resp['message']['content']
    chat_messages.append({
      'role': 'translator',
      'content': resp_text
    })

    if len(chat_messages) > 31:
      del chat_messages[1:len(chat_messages) - 30]

    return resp_text
  except Exception as e:
    print(f"Error: An unexpected error occurred while translating: {e}")
    return ""

def main():
  global chat_messages
  try:
    resp = requests.get(f"{SERVER_URL}")
    resp.raise_for_status()
  except Exception as e:
    print(f"Error: Cannot connect to Ollama server: {e}")
    exit(1)

  subs_dir = os.path.join(os.path.dirname(__file__), 'subs')
  files = os.listdir(subs_dir)
  total_files = len(files)

  for n, filename in enumerate(files):
    print(f"\nFile {filename} ({n + 1}/{total_files}):")
    filepath = os.path.join(subs_dir, filename)

    if not filepath.endswith(".srt"):
      print(f"Warning: File {filepath} is not an SRT file, skipping")
      continue

    new_subs: list[srt.Subtitle] = list()
    new_sub_id = 1
    with open(filepath, 'r') as file:
      subs = list(srt.parse(''.join(file.readlines())))
      total_subs = len(subs)

      prev_line = ""
      line_builder = ""
      start_sub = subs[0]
      for index, sub in enumerate(subs):
        progress = (index + 1) / total_subs * 100
        print(f"\rReformatting... {progress:.2f}% complete", end='')

        sub_lines = sub.content.split("\n")

        for line in sub_lines:
            line = line.strip()

            if (starts_with_hyphen(line) and prev_line.strip() and not ends_with_punctuation(prev_line.strip())):
              line = line[1:].strip()
              line_builder += " "
            elif (starts_with_hyphen(line)):
              line_builder += "\n"
            else:
              line_builder += " "

            line_builder += line
            prev_line = line

        if (ends_with_punctuation(prev_line)):
          new_subs.append(srt.Subtitle(new_sub_id, start_sub.start, sub.end, line_builder.strip(), ""))

          new_sub_id += 1
          line_builder = ""
          prev_line = ""
          start_sub = subs[-1] if sub == subs[-1] else subs[subs.index(sub) + 1]
    print()

    total_new_subs = len(new_subs)
    for index, sub in enumerate(new_subs):
      progress = (index + 1) / total_new_subs * 100
      print(f"\rTranslating... {progress:.2f}% complete", end='')

      translated_content = ""
      if "\n" in sub.content:
        sub_lines = sub.content.split("\n")
        for line in sub_lines:
          if line.startswith("-"):
            translated_content += "- "
            line = line[1:]

          line = line.strip()
          translated_content += translate(line).strip() + "\n"
      elif ". " in sub.content:
        sub_lines = sub.content.split(". ")
        for line in sub_lines:
          line = line.strip().strip(".") + "."
          translated_content += translate(line).strip() + " "
      else:
        translated_content = translate(sub.content)

      #translated_content = translate(sub.content)

      sub.content += f"\n<span style=\"color: yellow;\"><i>{translated_content.replace(">", "").replace("<", "").strip()}</i></span>"
    print()

    with open(filepath, 'w') as new_file:
      new_file.write(srt.compose(new_subs))
  
    chat_messages = EMPTY_CHAT_MESSAGES


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass