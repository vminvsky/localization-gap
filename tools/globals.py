import json
import os

def load_country_globals():
    country_lang_json_path = os.path.join(os.path.dirname(__file__),"json_data/country_lang.json")
    with open(country_lang_json_path,"r") as f:
        country_lang=json.load(f)
    for country in country_lang:
        country_name = country["country"]
        for lang in country["languages"]:
            if lang["most_common"]:
                headers=lang["translations"]
                #LANG_HEADER_MAP[lang["language"]] = (headers["Question:"], headers["Options:"], headers["Answer:"])
                COUNTRY_HEADER_MAP[country_name] = (headers["Question:"], headers["Options:"], headers["Answer:"])
                COUNTRY_LANGUAGE_MAP[country_name] = lang["language"]
def load_lang_headers():
    lang_headers_path = os.path.join(os.path.dirname(__file__),"json_data/lang_headers.json")
    with open(lang_headers_path,"r") as f:
        lang_headers = json.load(f)
    lang_header_map = {}
    for lang in lang_headers:
        ent = lang_headers[lang]
        lang_header_map[lang] = (ent["question"], ent["option"], ent["answer"])
    return lang_header_map
def load_demonstrations():
    demonstration_path = os.path.join(os.path.dirname(__file__),"json_data/demonstrations.json")
    with open(demonstration_path,"r") as f:
        demonstrations = json.load(f)
    return demonstrations

def load_prompt_lang_map():
    prompt_lang_map_path = os.path.join(os.path.dirname(__file__),"json_data/instruction_prompts_cult_bench.json")
    
    with open(prompt_lang_map_path,"r") as f:
        prompt_lang_list = json.load(f)

    prompt_lang_map = {}

    for ent in prompt_lang_list:
        lang = ent["lang"]
        prompt_lang_map[lang] = ent["prompt"]
    
    return prompt_lang_map


def reload_globals():
    global DEMONSTRATIONS, PROMPT_LANG_MAP, LANG_HEADER_MAP, COUNTRY_HEADER_MAP, COUNTRY_LANGUAGE_MAP
    DEMONSTRATIONS = load_demonstrations()
    PROMPT_LANG_MAP = load_prompt_lang_map()
    LANG_HEADER_MAP = load_lang_headers()
    COUNTRY_HEADER_MAP = {}
    COUNTRY_LANGUAGE_MAP = {}

    load_country_globals()

DEMONSTRATIONS = load_demonstrations()
PROMPT_LANG_MAP = load_prompt_lang_map()

LANG_HEADER_MAP = load_lang_headers()
COUNTRY_HEADER_MAP = {}
COUNTRY_LANGUAGE_MAP = {}

DEFAULT_HEADER = ("Question:","Options:","Answer:")

load_country_globals()
