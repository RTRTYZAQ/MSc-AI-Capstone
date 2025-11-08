import json

with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/fine-tuning/llava_finetune_1.json', "r") as f:
    dataset = json.load(f)

for data in dataset:
    data["image"] = f'/Datasets/Dermnet/1/{data["image"]}'

with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/fine-tuning/llava_finetune_1.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)