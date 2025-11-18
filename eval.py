import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
import math
import re


def inference_llava(model_path2, model_path1, model_base, data):
    disable_torch_init()
    model_path = os.path.expanduser(model_path2)
    model_name = get_model_name_from_path(model_path2)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path2, model_path1, model_base, model_name)

    model.to(torch.float16)

    qs = data["conversations"][0]["value"]
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(data["image"]).convert('RGB')

    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if 0.2 > 0 else False,
            temperature=1.0,
            top_p=0.9,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def standardized_json(json_str):
    json_str = re.sub(r'^\s*```(json)?\s*\n?', '', json_str)
    json_str = re.sub(r'\s*```\s*$', '', json_str)
    json_str = json_str.strip()

    return json_str


with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval/ground_truth.json', 'r') as f:
    dataset = json.load(f)

save = []

for data in dataset:
    save.append(standardized_json(inference_llava('/Checkpoints/LLaVA/llava-Capstone-lora-stage2', '/Checkpoints/LLaVA/llava-Capstone-lora-stage1', 'liuhaotian/llava-v1.5-13b', data)))


with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval.json', 'w', encoding='utf-8') as f:
    json.dump(save, f, ensure_ascii=False, indent=2)