from dataclasses import dataclass, field 
from llm_wrappers.models import OpenAIModel, HumanMessage
from llm_wrappers.important_utils import get_api_key
from pathlib import Path

from translate_utils import read_txt

PROMPT_PATH = Path(__file__).parent.parent / "data" / "prompts"

@dataclass 
class Translation:
    model_provider: str = field(default=OpenAIModel)
    model_name: str = field(default='gpt-4o-2024-08-06')

    src_lang: str = field(default=None)
    target_lang: str = field(default=None)

    model: OpenAIModel = field(default=None)
    system_prompt: str = field(default=None)
    system_prompt_path: str = field(default=PROMPT_PATH / 'translate_prompt.txt')

    def __post_init__(self):
        self.system_prompt = read_txt(self.system_prompt_path)
        self.model = self.model_provider(model_name=self.model_name, model_key=get_api_key())

    def __call__(self, text):
        temp_prompt = [HumanMessage(self.system_prompt.format(text=text, src_lang=self.src_lang, tar_lang=self.target_lang))]
        return self.model(temp_prompt)
    
@dataclass 
class RemoveCountryReference:
    model_provider: str = field(default=OpenAIModel)
    model_name: str = field(default='gpt-4o-2024-08-06')

    model: OpenAIModel = field(default=None)
    system_prompt: str = field(default=None)
    system_prompt_path: str = field(default=PROMPT_PATH / 'remove_country_reference.txt')

    def __post_init__(self):
        self.system_prompt = read_txt(self.system_prompt_path)
        
        self.model = self.model_provider(model_name=self.model_name, model_key=get_api_key())

    def __call__(self, text):
        temp_prompt = [HumanMessage(self.system_prompt.format(text=text))]
        return self.model(temp_prompt)

if __name__ == '__main__':
    # test to see this works 
    t_model = Translation(src_lang='en', target_lang='ru')
    translation = t_model('Translate this text woohoo')
    print(translation)