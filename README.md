# LLMSRTTranslator
Translate (multiple) SRT files with an LLM to any language you like.

# Requirements
- python 3.12 (may also work with older versions)
- libraries: `pip install -r requirements.txt`
- Access to a running [ollama](https://ollama.com/) server with a LLM already installed.
- subtitle files in SRT format

# Simple usage
- put your SRT(s) in the `subs` directory. The files must end in `.srt`
- open `translator.py` and edit the config constants according to your setup
- run the translator using `python translator.py`

# Advanced usage ([opensubtitles.org](https://www.opensubtitles.org/))
You can also download a whole season of a series from one specific uploader from opensubtitles.net:
- download the subtitles zip file, e.g. using a URL like this one: https://www.opensubtitles.org/en/download/s/sublanguageid-eng/uploader-mrtinkles/pimdbid-1091909/season-X
- put the downloaded zip in the `subs` directory
- run the `unpack.sh` bash script: `chmod +x unpack.sh && ./unpack.sh`
- now you can run the translator as described above.