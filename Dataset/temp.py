import json
import random

with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/fine-tuning/llava_finetune_2.json', "r") as f:
    dataset = json.load(f)

save = []

for data in dataset:
    if len(save) < 10:
        rand = random.randint(0, len(dataset))

        from PIL import Image
        img = Image.open(dataset[rand]['image'])
        img.save(f"/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval/{dataset[rand]['image'].split('/')[-1]}")

        dataset[rand]["image"] = f"/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval/{dataset[rand]['image'].split('/')[-1]}"

        save.append(dataset[rand])
    else:
        break

with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval/ground_truth.json', 'w', encoding='utf-8') as f:
    json.dump(save, f, ensure_ascii=False, indent=2)