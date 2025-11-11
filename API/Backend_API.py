from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
import uvicorn
import base64
import io
from PIL import Image

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






app = FastAPI()


# ------------ Pydantic Data Models ------------
class MessageContent(BaseModel):
    image: Optional[str] = None   # 前端传 image_path 或 base64
    text: Optional[str] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

class ConversationRequest(BaseModel):
    model: str
    messages: List[Message]
    top_p: float = 0.8
    temperature: float = 0.7


# ------------ 模型推理核心 ------------
def run_vlm_inference(model_name, model_path1, model_path2, model_base, image_path, user_prompt, top_p, temperature):
    """
    ✨ 这是你的 VLM 核心推理函数 ✨
    你可以把自己的 VLM pipeline 接进来，比如：
    result = your_model.generate(...)
    """

    disable_torch_init()
    model_path = os.path.expanduser(model_path2)
    model_name = get_model_name_from_path(model_path2)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path2, model_path1, model_base, model_name)

    model.to(torch.float16)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(image_path).convert('RGB')

    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if 0.2 > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


# ------------ API Entry ------------
@app.post("/multimodal/conversation")
async def multimodal_conversation(req: ConversationRequest):

    # --- 从 messages 中解析出 system prompt + image + text ---
    system_prompt = None
    image_path = None
    user_prompt = None

    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content

        if msg.role == "user":
            # content 为 list
            for block in msg.content:
                if block.image:
                    image_path = block.image
                if block.text:
                    user_prompt = block.text

    if user_prompt is None:
        raise HTTPException(status_code=400, detail="User prompt missing.")

    # --- 调用你的 VLM 推理 ---
    result = run_vlm_inference(
        model_name=req.model,
        model_path1="/Checkpoints/LLaVA/llava-Capstone-lora-stage1",
        model_path2="/Checkpoints/LLaVA/llava-Capstone-lora-stage2",
        model_base="liuhaotian/llava-v1.5-13b",
        image_path=image_path,
        user_prompt=user_prompt,
        top_p=req.top_p,
        temperature=req.temperature
    )

    # --- 包装成前端所需 response 结构 ---
    return {
        "status_code": 200,
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result
                    }
                }
            ]
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
