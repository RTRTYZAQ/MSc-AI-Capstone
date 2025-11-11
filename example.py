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


def inference_llava(model_path2, model_path1, model_base):
    disable_torch_init()
    model_path = os.path.expanduser(model_path2)
    model_name = get_model_name_from_path(model_path2)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path2, model_path1, model_base, model_name)

    model.to(torch.float16)

    qs = "<image>\nThe image is from the dermatology category: Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions.\nPlease generate a diagnostic report based on the skin images uploaded by the patient and the relevant knowledge about the skin condition provided. \nRelevant knowledge: {\n  \"disease\": [\n    {\n      \"question\": \"What is Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions?\",\n      \"confidence\": 0.9936,\n      \"text\": \"# Actinic keratosis Actinic keratosis ( AK ), sometimes called solar keratosis or senile keratosis , [1][2] is a pre-cancerous [3] area of thick, scaly, or crusty skin. [4][5] Actinic keratosis is a disorder ( -osis ) of epidermal keratinocytes that is induced by ultraviolet (UV) light exposure ( actin-). [6] These growths are more common in fair-skinned people and those who are frequently in the sun. [7] They are believed to form when skin gets damaged by UV radiation from the sun or indoor tanning beds, usually over the course of decades. Given their pre-cancerous nature, if left untreated, they may turn into a type of skin cancer called squamous cell carcinoma. [5] Untreated lesions have up to a 20% risk of progression to squamous cell carcinoma, [8] so treatment by a dermatologist is recommended.\"\n    },\n    {\n      \"question\": \"What is Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions?\",\n      \"confidence\": 0.9086,\n      \"text\": \"## Diagnosis Physicians usually diagnose actinic keratosis by doing a thorough physical examination, through a combination of visual observation and touch. However a biopsy may be necessary when the keratosis is large in diameter, thick, or bleeding, in order to make sure that the lesion is not a skin cancer. Actinic keratosis may progress to invasive squamous cell carcinoma (SCC) but both diseases can present similarly upon physical exam and can be difficult to distinguish clinically. [6] Histological examination of the lesion from a biopsy or excision may be necessary to definitively distinguish actinic keratosis from in situ or invasive SCC. [6] In addition to SCCs, actinic keratoses can be mistaken for other cutaneous lesions including seborrheic keratoses, basal cell carcinoma, lichenoid keratosis,\"\n    }\n  ],\n  \"treatment\": [\n    {\n      \"question\": \"What are the recommended treatment options for Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions?\",\n      \"confidence\": 0.9936,\n      \"text\": \"If clinical examination findings are not typical of actinic keratosis and the possibility of in situ or invasive squamous cell carcinoma (SCC) cannot be excluded based on clinical examination alone, a biopsy or excision can be considered for definitive diagnosis by histologic examination of the lesional tissue. [11] Multiple treatment options for actinic keratosis are available. Photodynamic therapy (PDT) is one option for the treatment of numerous actinic keratosis lesions in a region of the skin, termed field cancerization. [12] It involves the application of a photosensitizer to the skin followed by illumination with a strong light source. Topical creams, such as 5-fluorouracil or imiquimod, may require daily application to affected skin areas over a typical time course of weeks. [13]\"\n    },\n    {\n      \"question\": \"What are the recommended treatment options for Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions?\",\n      \"confidence\": 0.9502,\n      \"text\": \"# Retinoids Topical retinoids have been studied in the treatment of actinic keratosis with modest results, and the American Academy of Dermatology does not recommend this as first-line therapy. [66] Treatment with adapalene gel daily for 4 weeks, and then twice daily thereafter for a total of nine months led to a significant but modest reduction in the number of actinic keratoses compared to placebo; it demonstrated the additional advantage of improving the appearance of photodamaged skin. [67] Topical tretinoin is ineffective as treatment for reducing the number of actinic keratoses. [25] For secondary prevention of actinic keratosis, systemic, low-dose acitretin was found to be safe, well tolerated and moderately effective in chemoprophylaxis for skin cancers in kidney transplant patients. [68] Acitretin is a viable treatment option for organ transplant patients according to expert opinion. [46] ## Tirbanibulin\"\n    }\n  ]\n}\nPlease output strictly in JSON format, without any extra explanation:\n{{\n  \"Disease Name\": \"<predicted disease name>\",\n  \"Symptom Description\": \"<simple description of patient's skin condition symptoms>\",\n  \"Treatment Plan Recommendation\": \"<given this patient's skin condition and the relevant knowledge about this dermatological disorder, please provide your treatment plan recommendation.>\"\n}}"

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open('ctcl-118.jpg').convert('RGB')

    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if 0.2 > 0 else False,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print(f"\n\n{outputs}")


inference_llava('/Checkpoints/LLaVA/llava-Capstone-lora-stage2', '/Checkpoints/LLaVA/llava-Capstone-lora-stage1', 'liuhaotian/llava-v1.5-13b')