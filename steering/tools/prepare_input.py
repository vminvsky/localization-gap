from tools.globals import COUNTRY_HEADER_MAP, DEFAULT_HEADER, DEMONSTRATIONS, LANG_HEADER_MAP, PROMPT_LANG_MAP
from datasets import load_dataset
import pandas as pd

def prepare_q_prompt(question, translate=False):
    """
    Prepare questions for processing by LLM
    """
    if translate:
        question_header,option_header,_ = LANG_HEADER_MAP[question["lang"]]
    else:
        question_header,option_header,_ = DEFAULT_HEADER
    
    options_str = "\n".join([f"{i+1}. {option}" for i, option in enumerate(question["options"])])
    return f"{question_header}\n{question['question']}\n{option_header}\n{options_str}"

def prepare_ans_prompt(question, translate=False):
    """
    Prepare answers for processing by LLM
    """
    if translate:
        _,_,answer_header = COUNTRY_HEADER_MAP.get(question["country"], DEFAULT_HEADER)
    else:
        _,_,answer_header = DEFAULT_HEADER

    return f"{answer_header.strip()}\n{question['answer']+1}"

def prepare_messages(prompt_data, include_demonstrations=True, add_answer_header=False, add_instruct=False, demonst_limit=-1):
    lang = prompt_data["lang"]
    messages = []
    if include_demonstrations:
        messages.extend(DEMONSTRATIONS.get(lang, []))
        if demonst_limit > 0:
            messages = messages[:demonst_limit*2]
        for m in messages:
            if m["role"] == "assistant":
                m["content"] = m["content"].split(".")[0]

    messages.append({"role":"user","content":prompt_data["prompt"]})

    if add_instruct:
        new_message = {"role":"user","content":PROMPT_LANG_MAP[lang].strip() + "\n" + messages[0]["content"]}
        messages[0] = new_message
    if add_answer_header:
        messages.append({"role":"assistant","content":LANG_HEADER_MAP[lang][2].strip()})
    
    return messages

def base_chat_template(messages):
    str_to_return = ""
    for m in messages:
        str_to_return += f"{m['content']}\n"
    return str_to_return

def messages_to_str(messages, tokenizer, instruction_model=False):
    if type(messages) == str:
        messages = [{"role":"user", "content":messages}]
    if instruction_model:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return base_chat_template(messages)


def get_batches(dataset, tokenizer, batch_size=8, instruction_model=False):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        idxs = [e["idx"] for e in batch]

        if instruction_model:
            texts = [tokenizer.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=True) for e in batch]
        else:
            texts = [base_chat_template(e["messages"]) for e in batch]
        batches.append((idxs,texts))
    return batches

def get_original_culturebench():
    dataset = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy")
    dataset_df = pd.DataFrame(dataset["test"])

    def get_prompt(row):
        prompt="To answer the following multiple-choice question, you should choose one option only among A,B,C,D. Instruction: You must select one option among A,B,C,D. Do not output any other things."
        prompt += f" {row['prompt_question']}\nA. {row['prompt_option_a']}\nB. {row['prompt_option_b']}\nC. {row['prompt_option_c']}\nD. {row['prompt_option_d']}"
        return prompt

    dataset_df["prompt"] = dataset_df.apply(get_prompt, axis=1)
    dataset_df["idx"] = dataset_df.index
    dataset_df["messages"] = dataset_df.apply(lambda row:[{"role":"user","content":row["prompt"]}], axis=1)
    return dataset_df

def get_created_dataset(name, add_demonstrations=False, instruction_model=False, demonst_limit=-1):
    dataset_df = pd.read_csv("data/{name}.csv".format(name=name))
    dataset_df.dropna(subset=["prompt"], inplace=True)
    dataset_df["idx"] = dataset_df.index
    dataset_df["messages"] = dataset_df.apply(lambda row:prepare_messages(row,
                                                                        add_demonstrations,
                                                                        not (instruction_model),
                                                                        instruction_model,
                                                                        demonst_limit=demonst_limit),
                                                                        axis=1)
    return dataset_df

def get_name_dataset(name_df, add_instruct=False, include_demonstrations=True, add_answer_headers=False):
    en_instruct = "Instruction: You must select one option among 1,2. Do not output any other things."
    tr_instruct = "Talimat: 1,2 arasından bir seçenek seçmelisiniz. Başka bir şey çıkarmayın."

    en_demonstration = [{"role":"user", "content":"Question:\nWho am I?\nOptions:\n1. Human\n2. Robot"}, 
     {"role":"assistant", "content":"Answer:\n1"}]
    tr_demonstration = [{"role":"user", "content":"Soru:\nBen kimim?\nSeçenekler:\n1. İnsan\n2. Robot"}, 
     {"role":"assistant", "content":"Cevap:\n1"}]
    all_data = []
    for idx, row in name_df.iterrows():
        "rank_us,name_us,gender,rank_tr,name_tr,answer_us,answer_tr"
        data= {"idx": idx, 
               "name_en": row["name_us"],
               "name_tr": row["name_tr"],
               "gender": row["gender"],
               "rank_us": row["rank_us"],
                "rank_tr": row["rank_tr"],
               "answer_en": row["answer_us"], 
               "answer_tr": row["answer_tr"]}
        for suffix in ["en", "tr", "en_tur_hint", "tr_tur_hint", "en_us_hint", "tr_us_hint", "en_task", "tr_task"]:
            messages = [{"role":"user", "content":row[f"prompt_{suffix}"]}]

            if add_answer_headers:
                if suffix.startswith("en"):
                    messages.append({"role":"assistant", "content":"Answer:"})
                elif suffix.startswith("tr"):
                    messages.append({"role":"assistant", "content":"Cevap:"})

            if include_demonstrations:
                if suffix.startswith("en"):
                    messages = en_demonstration + messages
                elif suffix.startswith("tr"):
                    messages = tr_demonstration + messages
            
            if add_instruct:
                if suffix.startswith("en"):
                    messages[0]["content"] = en_instruct + "\n" + messages[0]["content"]
                elif suffix.startswith("tr"):
                    messages[0]["content"] = tr_instruct + "\n" + messages[0]["content"]
            data["messages_"+suffix] = messages
        all_data.append(data)
    return all_data

def get_name_dataset_v2(name_df, add_instruct=False, include_demonstrations=True, add_answer_headers=False):
    en_instruct = "You must select one option and answer. Then explain your guess."
    tr_instruct = "Seçeneklerden birini seçip cevap vermek zorundasın. Sonra tahminini açıkla."

    en_demonstration = [{"role":"user", "content":"Question:\nWho am I?\nOptions:\nHuman\nRobot"}, 
     {"role":"assistant", "content":"Answer:\nHuman"}]
    tr_demonstration = [{"role":"user", "content":"Soru:\nBen kimim?\nSeçenekler:\nİnsan\nRobot"}, 
     {"role":"assistant", "content":"Cevap:\nİnsan"}]
    all_data = []
    for idx, row in name_df.iterrows():
        "rank_us,name_us,gender,rank_tr,name_tr,answer_us,answer_tr"
        data= {"idx": idx, 
               "name_en": row["name_us"],
               "name_tr": row["name_tr"],
               "gender": row["gender"],
               "rank_us": row["rank_us"],
                "rank_tr": row["rank_tr"],
               "answer_en": str(int(row["answer_us"])), 
               "answer_tr": str(int(row["answer_tr"]))}
        for suffix in ["en", "tr", "en_tur_hint", "tr_tur_hint", "en_us_hint", "tr_us_hint", "en_task", "tr_task"]:
            prompt = row[f"prompt_{suffix}"].replace("1. ", "").replace("2. ", "")
            messages = [{"role":"user", "content":prompt}]

            if add_answer_headers:
                if suffix.startswith("en"):
                    messages.append({"role":"assistant", "content":"Answer:"})
                elif suffix.startswith("tr"):
                    messages.append({"role":"assistant", "content":"Cevap:"})

            if include_demonstrations:
                if suffix.startswith("en"):
                    messages = en_demonstration + messages
                elif suffix.startswith("tr"):
                    messages = tr_demonstration + messages
            
            if add_instruct:
                if suffix.startswith("en"):
                    messages[0]["content"] = messages[0]["content"].replace("Question:",f"Question:\n{en_instruct}")
                elif suffix.startswith("tr"):
                    messages[0]["content"] = messages[0]["content"].replace("Soru:",f"Soru:\n{tr_instruct}")
            data["messages_"+suffix] = messages
        all_data.append(data)
    return all_data

def get_city_dataset(city_df, add_instruct=False, include_demonstrations=True, add_answer_headers=False):
    en_instruct = "You must select one option and answer. Then explain your guess."
    tr_instruct = "Seçeneklerden birini seçip cevap vermek zorundasın. Sonra tahminini açıkla."

    en_demonstration = [{"role":"user", "content":"Question:\nWho am I?\nOptions:\nHuman\nRobot"}, 
     {"role":"assistant", "content":"Answer:\nHuman"}]
    tr_demonstration = [{"role":"user", "content":"Soru:\nBen kimim?\nSeçenekler:\nİnsan\nRobot"}, 
     {"role":"assistant", "content":"Cevap:\nİnsan"}]
    
    all_data = []

    for idx, row in city_df.iterrows():
        row = row.to_dict()
        

