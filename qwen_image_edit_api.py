import base64
import requests
import json
import mimetypes

API_KEY = "your api key"   # ⚠️ 不要泄露
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def encode_image_base64(file_path):
    mime = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    with open(file_path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def edit_image_plus(input_path, prompt, output_jpg):
    print("Encoding image...")
    img_b64 = encode_image_base64(input_path)

    payload = {
        "model": "qwen-image-edit-plus",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": img_b64},
                        {"text": prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,                    # 生成 1 张图
            "negative_prompt": "",
            "prompt_extend": True,
            "watermark": False
        }
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print("Sending request to qwen-image-edit-plus ...")
    resp = requests.post(API_URL, json=payload, headers=headers)

    print("Raw response:")
    print(resp.text)

    try:
        data = resp.json()
    except:
        print("❌ JSON 解析失败")
        return

    if "output" not in data:
        print("❌ API Error:", data)
        return

    # 取到图片 URL
    img_url = data["output"]["choices"][0]["message"]["content"][0]["image"]
    print(f"Image URL (24h valid): {img_url}")

    # 下载保存
    print("Downloading generated image...")
    img_data = requests.get(img_url).content
    with open(output_jpg, "wb") as f:
        f.write(img_data)

    print("✅ Saved:", output_jpg)


# ----------- 运行示例 ----------
edit_image_plus(
    input_path="/root/autodl-tmp/tools/wangzheng.jpg",
    prompt="恢复全身，补全四肢，真实写实风格",
    output_jpg="/root/autodl-tmp/tools/qwen_edit_output.png"
)
