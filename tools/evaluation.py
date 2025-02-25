import torch as t
from torch import Tensor
import torch.nn.functional as F
import pandas as pd
import re
from collections import Counter



def extract_answers(results_df):
    def extract_number(text):
        pattern = re.compile(r"([0-9]+)")
        match = pattern.search(text)
        if match:
            return int(match.group())-1
        return -1
    results_df["extracted_answer"] = results_df["out"].apply(lambda x: extract_number(x))
    results_df["is_valid"] = (results_df["extracted_answer"]>=0)
    results_df["is_correct"] = (results_df["extracted_answer"]==results_df["answer"])
    return results_df

def calculate_accuracy(results_df):
    return (results_df["extracted_answer"]==results_df["answer"]).sum() / len(results_df)

def print_accuracy_summary(results_df):
    hint = [True,False]
    translated = [True,False]
    for h in hint:
        for l in translated:
            ans = results_df.query(f"hint == {h} and translated == {l}")
            valid_ans = ans.query(f"extracted_answer != -1")
            invalid_ans = ans.query(f"extracted_answer == -1")
            correct_count = (ans["extracted_answer"] == ans["answer"]).sum()
            total_count = len(ans)
            print(f"Hint: {h}, Translated: {l}")
            print(f"Correct: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%) Correct (in valid): {correct_count}/{len(valid_ans)} ({correct_count/len(valid_ans)*100:.2f}%) Invalid: {len(invalid_ans)}")

def return_grouped_results(results_df):
    def merge_rows(group):
        new_row = {}
        for _, r in group.iterrows():
            suffix = ""
            
            if r['translated']:
                new_row["country"] = r['country']
                new_row["answer"] = r['answer']
                #new_row["answer_str"] = r["answer_str"]
                new_row["target_lang"] = r['lang']
                new_row["options_translated"] = r['options']
                suffix += "_translated"
            else:
                new_row["options"] = r['options']
            if r['hint']:
                suffix += "_hint"
            
            for col in ['question', 'prompt', 'messages', 'out', 'in', "extracted_answer"]:
                if col in r:
                    new_row[f"{col}{suffix}"] = r[col]
                else:
                    continue
        
        return new_row

    merged_df = pd.DataFrame(results_df.groupby('question_idx').apply(merge_rows).to_list()).reset_index(drop=True)
    return merged_df

def display_entries(selected_entries, columns=['idx', 'question_idx', 'country', 'prompt', 'answer', 'out']):
    print(f"Displaying {len(selected_entries)} entries:")
    for i, row in selected_entries.iterrows():
        for col in columns:
            if col in row:
                print(f"{col.capitalize()}: {row[col]}")
            else:
                print(f"{col.capitalize()}: N/A")
        print()

def load_name_guesses(file_path, check_for = "index", format="{0}"):
    def get_answer_type(row):
        if check_for=="index":
            ans_en = str(int(row["answer_en"]))
            ans_tr = str(int(row["answer_tr"]))
        else:
            ans_en = format.format(row["name_en"])
            ans_tr = format.format(row["name_tr"])

        for suffix in ["en", "tr", "en_tur_hint", "tr_tur_hint", "en_us_hint", "tr_us_hint", "en_task", "tr_task"]:
            row["ans_type_"+suffix] = None
            if ans_en in row["output_"+suffix] and ans_tr in row["output_"+suffix]:
                en_index = row["output_"+suffix].index(ans_en)
                tr_index = row["output_"+suffix].index(ans_tr)
                row["ans_type_"+suffix] = "en" if en_index < tr_index else "tr"
            elif ans_en in row["output_"+suffix]:
                row["ans_type_"+suffix] = "en"
            elif ans_tr in row["output_"+suffix]:
                row["ans_type_"+suffix] = "tr"

        return row
    result_df = pd.read_csv(file_path)
    result_df = result_df.apply(get_answer_type, axis=1)
    return result_df

def display_df(df, query=None, columns=None):
    df_to_display = df.copy()
    if query is not None:
        df_to_display = df_to_display.query(query)
    if columns is not None:
        df_to_display = df_to_display[columns]
    for i, row in df_to_display.iterrows():
        for col, val in row.items():
            print(f"{col}: {val}")
        print()

def get_name_ans_types(results_df, show="count"):
    # Get columns that start with 'ans_type_'
    ans_type_columns = [col for col in results_df.columns if col.startswith('ans_type_')]

    # Count frequencies of element occurrences in these columns
    for col in ans_type_columns:
        print(col)
        frequencies = Counter(results_df[col])
        if show == "count":
            print(frequencies)
        elif show == "percentage":
            total = sum(frequencies.values())
            for key, value in frequencies.items():
                print(f"{key}: {value/total*100:.2f}%")



def calculate_perplexity(logits: Tensor, target_tokens: Tensor) -> float:
    # Get the shape of the logits tensor
    _, _, vocab_size = logits.shape

    # Reshape logits and targets
    logits_flat = logits.view(-1, vocab_size)   # Shape: (8, 5)
    targets_flat = target_tokens.view(-1)       # Shape: (8)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')

    # Compute perplexity
    perplexity = t.exp(loss)

    return perplexity.item()


def get_answer_type_final(row, check_for = "index", format="{0}"):
    if check_for=="index":
        if row["source_id"] == "xculturebench":
            ans_en = "culturebench"
            ans_tr = str(int(row["ans_local_idx"])+1)

        else:
            ans_en = str(int(row["ans_west_idx"]))
            ans_tr = str(int(row["ans_local_idx"]))
    else:
        ans_en = format.format(row["ans_west"]).lower().strip()
        ans_tr = format.format(row["ans_local"]).lower().strip()
    
    if row["source_id"] == "xculturebench":
        ans_en = "culturebench"

    
    row["ans_type"] = "none"
    out = str(row["output"]).lower().strip()
    if ans_en in out and ans_tr in out:
        en_index = out.index(ans_en)
        tr_index = out.index(ans_tr)
        row["ans_type"] = "west" if en_index < tr_index else "local"
    elif ans_en in out:
        row["ans_type"] = "west"
    elif ans_tr in out:
        row["ans_type"] = "local"

    return row
