# Qwen_inpaint.py  —— 在 qwen_img 环境、本地 Qwen-Image-Edit-2509 上做补全

import os
import argparse

# 1) 关闭 Flash Attention，强制用 eager（防止 sdpa 乱七八糟）
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_FORCE_EAGER"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# 2) 打补丁：去掉 scaled_dot_product_attention 的 enable_gqa 参数
original_scaled_dot_product_attention = F.scaled_dot_product_attention

def patched_scaled_dot_product_attention(*args, **kwargs):
    # 移除 enable_gqa 参数（旧版 torch 不认识）
    kwargs.pop("enable_gqa", None)
    return original_scaled_dot_product_attention(*args, **kwargs)

F.scaled_dot_product_attention = patched_scaled_dot_product_attention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument(
        "--prompt",
        default="restore full body, extend to head-to-toe, complete limbs, photorealistic, same person",
        help="正向提示词",
    )
    parser.add_argument(
        "--negative_prompt",
        default="extra limbs, blurry, distorted body",
        help="反向提示词",
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="采样步数"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="随机种子"
    )
    args = parser.parse_args()

    model_path = "/root/autodl-tmp/Qwen-Image-Edit-2509"

    # 3) 只用第二块卡：通过 CUDA_VISIBLE_DEVICES 控制
    #    外面会把 CUDA_VISIBLE_DEVICES=1 传进来，这里直接用 cuda:0 即可
    device = "cuda"

    print(f"[Qwen-2509] loading pipeline from {model_path} ...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # 保持你 demo.py 的配置
    )

    pipe.set_progress_bar_config(disable=None)

    image = Image.open(args.input).convert("RGB")

    inputs = {
        "image": image,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": 4.0,
        "num_inference_steps": args.steps,
        "guidance_scale": 1.2,
        "num_images_per_prompt": 1,
        "generator": torch.manual_seed(args.seed),
    }

    print("[Qwen-2509] start inference ...")
    with torch.inference_mode():
        out = pipe(**inputs)

    img = out.images[0]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    img.save(args.output)

    print("[Qwen-2509] done, saved to:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
