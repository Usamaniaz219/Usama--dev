import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import json
from numwords_to_nums.numwords_to_nums import NumWordsToNum
import os

# Load tokenizer and model
# model_path = "results/checkpoint-5250"
# model_path = "13_oct_2025_saved_checkpoints/checkpoint-3500"
# model_path = "results/checkpoint-5310"
# model_path = "results/checkpoint-6018"
model_path = "final_trained_models/21_oct_trained_model_final/checkpoint-5310"

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

# Load label mapping
with open(f"{model_path}/id_to_label.json", "r") as f:
    id_to_label = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True
    )

    # print(inputs)

    offset_mapping = inputs.pop("offset_mapping")[0]  # shape: (seq_len, 2)

    special_tokens_mask = inputs.pop("special_tokens_mask")[0]
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    input_ids = inputs["input_ids"][0].cpu().numpy()
    
    results = []

    for idx, (pred_id, input_id, offset, is_special) in enumerate(zip(predictions, input_ids, offset_mapping, special_tokens_mask)):
        if is_special or offset[0] == offset[1]:
            continue  # Skip [CLS], [SEP], and padding tokens

        token = tokenizer.convert_ids_to_tokens(int(input_id))
        label = id_to_label[str(pred_id)]
        start, end = offset.tolist()

        results.append({
            "token": token,
            "label": label,
            "start": start,
            "end": end
        })
    
    return results

def entity_extraction(predicted_tokens):
    entities = []
    current_tokens = []
    current_label = None
    start_index = None

    for token_info in predicted_tokens:
        label = token_info["label"]
        token = token_info["token"]
        start = token_info["start"]
        end = token_info["end"]

        if label != 'O':
            if current_label == label:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "text": tokenizer.convert_tokens_to_string(current_tokens).strip(),
                        "label": current_label,
                        "start": start_index,
                        "end": prev_end
                    })
                current_tokens = [token]
                current_label = label
                start_index = start
        else:
            if current_tokens:
                entities.append({
                    "text": tokenizer.convert_tokens_to_string(current_tokens).strip(),
                    "label": current_label,
                    "start": start_index,
                    "end": prev_end
                })
                current_tokens = []
                current_label = None
                start_index = None
        prev_end = end  # keep track of previous token's end

    if current_tokens:
        entities.append({
            "text": tokenizer.convert_tokens_to_string(current_tokens).strip(),
            "label": current_label,
            "start": start_index,
            "end": prev_end
        })

    return entities



def numwordTonums(text):
    num = NumWordsToNum()
    result = num.numerical_words_to_numbers(text,convert_operator=False,calculate_mode=False,evaluate=False)
    return result

def read_and_process_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return [process_line_to_tuple(line) for line in lines if line.strip()]



def process_line_to_tuple(line):
    line = line.strip().rstrip('.')
    if line:
        return (line.strip())
   
    

def run_entity_extraction(test_sentences, output_path):
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for test_sentence in test_sentences:
            token_predictions = predict(test_sentence)
            entities = entity_extraction(token_predictions)

            out_file.write(f"Test Sentence: {test_sentence}\n")
            out_file.write("Extracted Entities with Positions:\n")

            if entities:
                for ent in entities:
                    out_file.write(f"{ent}\n")
            else:
                out_file.write("[]\n")

            out_file.write("######################################################\n")
            out_file.write("######################################################\n\n")
    



if __name__ == "__main__":

    input_dir = "/media/usama/SSD/Usama_dev_ssd/name_entity_recognition_/bert_based_NER/src/test_parking_data_27_oct_2025/"
    output_dir = "/media/usama/SSD/Usama_dev_ssd/name_entity_recognition_/bert_based_NER/src/parking_sample_results_27_oct_2025/"

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir,filename)
            city_name = filename.replace(".txt","")
            # with open(file_path,"r",encoding="utf-8") as f:
            #     parsed_txt_file = f.read()


            output_file_path = os.path.join(output_dir, f"{city_name}.txt")

            tuples = read_and_process_file(file_path)
            # print("tuples",tuples)
            # print("length of tuples",len(tuples))
            run_entity_extraction(tuples, output_file_path)







