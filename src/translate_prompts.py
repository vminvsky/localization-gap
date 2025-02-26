import os
import json
from translate import Translation

text = 'Hello, this is an absurdly simple translation'

stories = [
    "Write me a recipe for a local dish.",
    "Write me a short story about a boy and his kite.",
    "Write me a 100 word bedtime story for kids.",
    "Tell me about what happened during the titanic.",
    "Describe a day in the life of an astronaut on Mars.",
    "Explain photosynthesis as if I'm five years old.",
    "Write a letter from a pirate to his long-lost friend.",
    "Invent a new holiday and describe how people celebrate it.",
    "Tell me a joke that would make a robot laugh.",
    "Describe the feeling of standing at the edge of a cliff.",
    "Write a poem about a lonely lighthouse.",
    "Explain gravity without using scientific jargon.",
    "Create a dialogue between a cat and a dog arguing about dinner.",
    "Write a product review for an imaginary gadget.",
    "Describe a futuristic city 500 years from now.",
    "Tell me a legend about a magical forest.",
    "Explain how to build a sandcastle like a pro.",
    "Write a diary entry from the perspective of a dragon.",
    "Imagine you're a time traveler—describe your first day in the past.",
    "Give me instructions on how to be invisible for a day.",
    "Write a letter from Earth to an alien civilization.",
    "Describe a sunset without using the words 'red,' 'orange,' or 'yellow.'",
    "Tell me about a secret hidden inside an old library.",
    "Invent a sport that could be played on the moon."
]

langs = [
    {'target_lang': 'tr', 'suffix': "You are speaking with a Turkish person."},
    {'target_lang': 'de', 'suffix': "You are speaking with a German person."},
    {'target_lang': 'fr', 'suffix': "You are speaking with a French person."},
    {'target_lang': 'en', 'suffix': "You are speaking with an American person."},
    {'target_lang': 'ru', 'suffix': "You are speaking with a Russian person."},
]

for lang in langs:
    translate_model = Translation(src_lang='en', target_lang=lang['target_lang'])

    for run_with_suffix in [True, False]:
        if run_with_suffix:
            lang['suffix'] = lang['suffix']
            fname = f'{lang["target_lang"]}'
        else:
            lang['suffix'] = None
            fname = f'{lang["target_lang"]}'
        save_path = f'data/open_ended_generation/{fname}_{run_with_suffix}.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        lang_data = []
        for story in stories:
            if run_with_suffix:
                story = f'{story} {lang["suffix"]}'
            
            if lang['target_lang'] == 'en':
                lang_data.append({'story': story, 'translation': story, 'lang': lang['target_lang']})
            else:
                t_text = translate_model(story)
                lang_data.append({'story': story, 'translation': t_text, 'lang': lang['target_lang']})
        with open(save_path, 'w') as f:
            json.dump(lang_data, f, indent=4, ensure_ascii=False)
