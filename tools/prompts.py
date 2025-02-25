from tools.globals import LANG_HEADER_MAP, PROMPT_LANG_MAP


def format_v1(row):
    instructions = {
        "English": "You must select one option and answer. First, state the selected option in full, then explain your guess.",
        "Turkish": "Seçeneklerden birini seçip cevap vermek zorundasın. Seçtiğin seçeneği tam olarak ver ve sonra tahminini açıkla.",
        "French" : "Vous devez sélectionner une option et répondre. D'abord, indiquez l'option sélectionnée en entier, puis expliquez votre supposition.",
        "Russian": "Вы должны выбрать один вариант и ответить. Сначала укажите выбранный вариант полностью, затем объясните своё предположение.",
        "Bengali": "আপনাকে একটি বিকল্প নির্বাচন করে উত্তর দিতে হবে। প্রথমে নির্বাচিত বিকল্পটি পূর্ণভাবে উল্লেখ করুন, তারপর আপনার অনুমান ব্যাখ্যা করুন।",
    }

    instruct = instructions[row["lang"]]
    question = row["question"]
    q_header, o_header, a_header = LANG_HEADER_MAP[row["lang"]]
    options = "\n".join(row["options"])

    return f"{q_header}{instruct}\n{question}\n\n{options}"

def format_v2(row):
    instructions = {
        "English": "You must select one option and answer. First, state the selected option in full, then explain your guess.",
        "Turkish": "Seçeneklerden birini seçip cevap vermek zorundasın. Seçtiğin seçeneği tam olarak ver ve sonra tahminini açıkla.",
        "French" : "Vous devez sélectionner une option et répondre. D'abord, indiquez l'option sélectionnée en entier, puis expliquez votre supposition.",
        "Russian": "Вы должны выбрать один вариант и ответить. Сначала укажите выбранный вариант полностью, затем объясните своё предположение.",
        "Bengali": "আপনাকে একটি বিকল্প নির্বাচন করে উত্তর দিতে হবে। প্রথমে নির্বাচিত বিকল্পটি পূর্ণভাবে উল্লেখ করুন, তারপর আপনার অনুমান ব্যাখ্যা করুন।",
    }


    instruct = instructions[row["lang"]]
    question = row["question"]
    q_header, o_header, a_header = LANG_HEADER_MAP[row["lang"]]
    options = "\n".join(row["options"])

    return f"{q_header}{instruct}\n{question}\n{o_header}\n{options}"

def format_multi_choice(row):
    instruct = PROMPT_LANG_MAP[row["lang"]]
    question = row["question"]
    q_header, o_header, a_header = LANG_HEADER_MAP[row["lang"]]
    options = "\n".join([f"{str(i+1)}. {opt}" for i, opt in enumerate(row["options"])])
    return f"{q_header}{instruct}\n{question}\n{o_header}\n{options}"

def format_base(row):
    #instruct = PROMPT_LANG_MAP[row["lang"]]
    question = row["question"]
    
    q_header, o_header, a_header = LANG_HEADER_MAP[row["lang"]]
    options = "\n".join([f"{str(i+1)}. {opt}" for i, opt in enumerate(row["options"])])

    return f"{q_header}\n{question}\n{o_header}\n{options}\n{a_header}"