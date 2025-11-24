<h1 align="center">
  <img src="/assets/bucea.jpg">
  <br>
  Large Animatable Human Model + Qwen-Image-Edit-2509 Integration
</h1>

<p align="center">
  <strong>SiYu Liao<sup>*</sup></strong><br>
  Beijing University of Civil Engineering and Architecture
</p>

[![Project Website](https://img.shields.io/badge/🌐-Project_Website-blueviolet)](https://github.com/2179888515junjie/LHM-Qwen-image-edit-2509-Course-Design-Assignment)



## 📢 Latest Updates
**[November 24,2025]** I have started writing the readme for this course design<br>
**[November 23,2025]** I have successfully completed the deployment of the local Qwen-Imag-Edit-2059 model and distributed code generation based on 2 48GB GPU memory blocks<br>
**[November 19,2025]** I have successfully completed the code generation for LHM based on the Qwen Image API interface call model<br>



## ⚙️ Environment Configuration

This project is based on two independent virtual environments (LHM-base and qwen_img), for the following reasons:

- **LHM model relies on old versions diffusers / accelerate / mmcv / pytorch3d**
- **Qwen-Image-Edit-2509 Local inference relies on the latest versions diffusers / transformers / accelerate**
- The two rely on strong conflicts, so they must operate in different environments.

In addition, due to the large size of the Qwen Image-Edit-2509 local model (≈ 57 GB), it requires high graphics memory GPU support.

The actual operating environment of this project is as follows:

### 🖥️ Hardware environment (AutoDL cloud)
|Device | Model | Video Memory | Quantity
|------|------|------|------|
| GPU | NVIDIA vGPU-48GB | 49 GB | 2 |
|CUDA Version | CUDA 13.0 | - | Driver 580.76.05|
|CPU | AutoDL default configuration | - | -|
|System | Ubuntu 20.04/Miniconda | - | -|
>  💡 ** Explanation:**
>Due to being based on Autodl, some paths in the code contain Autodl tmp, and file names can be changed as needed
>LHM-1B/LHM-500M can run stably on both 48GB GPUs of this machine;   
>The Qwen-Imag-Edit-2509 (local version) inference requires at least 48GB of video memory on a single card to fully load the model.
>When using local inference, it occupies video memory data as shown in the following figure
  
<img src="./assets/xiancun.png" />
  

1. Clone repository
```bash
git clone https://github.com/2179888515junjie/LHM-Qwen-image-edit-2509-Course-Design-Assignment
cd LHM-Qwen-image-edit-2509-Course-Design-Assignment

```

2. Create Environment 1: LHM-base (main environment)
  
This environment is used to run:
>LHM model (1B/500M)
>SAM2 segmentation
>Video Analysis (Video2MotionPipeline)
>3DGS reconstruction and rendering
 
**Create environment**
```
conda create -n LHM_base python=3.10
conda activate LHM_base
```


Install LHM dependency (compatible with CUDA 11.8)

```
#CUDA 11.8 Environment
sh ./install_cu118.sh
pip install rembg

Or CUDA 12. x:

# cuda 12.1
sh ./install_cu121.sh
pip install rembg
```
Key library versions used in this environment:

> torch==2.3.0+cu118  
> diffusers==0.36.0.dev0  
> mmcv / mmpose / pytorch3d  
> SAM2 (pip install sam-2)  
> accelerate==0.25  
> numpy==1.23.5

This environment has been tested and verified with a 48GB GPU and CUDA 11.8, and can run LHM Pipeline stably.


3. Create Environment 2: qwen_img (for local Qwen-Imag-Edit-2509 inference)

**Create environment**
<span style="color: yellow">Strongly recommend using diffusers recommended by Qwen official</span>

```
conda create -n qwen_img python=3.10
conda activate qwen_img
```

Installation of Qwen-Imag-Edit-2509 requires dependencies for inference:
```
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url  https://download.pytorch.org/whl/cu118

#Latest diffusers (as requested by Qwen)
pip install git+ https://github.com/huggingface/diffusers

# transformers / accelerate
pip install transformers accelerate

pip install safetensors pillow
```
**If there is an error message 'TypeError: scaled_dotsproduct_mattenting() got an unexpected keyword argument' enable_gqa ''**
If you don't want to change the environment version, you can apply a patch to temporarily disable the parameters:
> import torch.nn.functional as F
> original_scaled_dot_product_attention = F.scaled_dot_product_attention  
> def patched_scaled_dot_product_attention(*args, **kwargs):  
>Remove the enable_gqa parameter
> kwargs.pop('enable_gqa', None)  
> return original_scaled_dot_product_attention(*args, **kwargs)   
> F.scaled_dot_product_attention = patched_scaled_dot_product_attention



###LHM Model Download


If you haven't downloaded the model, it will be automatically downloaded</span>


|Model | Training Data | Transformer Layers | ModelScope | HuggingFace | Inference Time | Required Input|
| :--- | :--- | :--- | :--- | :--- | :--- |:--- |
|LHM-MINI | 300K video data+5K 3D data | 2 | [ModelScope]（ https://modelscope.cn/models/Damo_XR_Lab/LHM-MINI ) |[huggingface]( https://huggingface.co/3DAIGC/LHM-MINI ）|1.41 seconds | Whole body and half body|
|LHM-500M | 300K video data+5K 3D data | 5 | [ModelScope]（ https://modelscope.cn/models/Damo_XR_Lab/LHM-500M ) |[huggingface]( https://huggingface.co/3DAIGC/LHM-500M ）|2.01 seconds | Whole body|
|LHM-500M-HF | 300K video data+5K 3D data | 5 | [ModelScope]（ https://modelscope.cn/models/Damo_XR_Lab/LHM-500M-HF ) |[huggingface]( https://huggingface.co/3DAIGC/LHM-500M-HF ）|2.01 seconds | Whole body and half body|
|LHM-1.0B | 300K video data+5K 3D data | 15 | [ModelScope]（ https://modelscope.cn/models/Damo_XR_Lab/LHM-1B ) |[huggingface]( https://huggingface.co/3DAIGC/LHM-1B ）|6.57 seconds | Whole body|
|LHM-1B-HF | 300K video data+5K 3D data | 15 | [ModelScope]（ https://modelscope.cn/models/Damo_XR_Lab/LHM-1B-HF ) |[huggingface]( https://huggingface.co/3DAIGC/LHM-1B-HF ）|6.57 seconds | Whole body and half body|



####Download from HuggingFace
```python
from huggingface_hub import snapshot_download 
# MINI Model
model_dir = snapshot_download(repo_id='3DAIGC/LHM-MINI', cache_dir='./pretrained_models/huggingface')
# 500M-HF Model
model_dir = snapshot_download(repo_id='3DAIGC/LHM-500M-HF', cache_dir='./pretrained_models/huggingface')
# 1B-HF Model
model_dir = snapshot_download(repo_id='3DAIGC/LHM-1B-HF', cache_dir='./pretrained_models/huggingface')
```

###Download prior model weights
```bash
#Download prior model weights
wget  https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar  
tar -xvf LHM_prior_model.tar 
```

###Action data preparation
LHM provides test action examples:
```bash
#Download prior model weights
wget  https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/motion_video.tar
tar -xvf ./motion_video.tar 
```

After downloading, the LHM file structure is as follows:
```bash
├── configs
│   ├── inference
│   ├── accelerate-train-1gpu.yaml
│   ├── accelerate-train-deepspeed.yaml
│   ├── accelerate-train.yaml
│   └── infer-gradio.yaml
├── engine
│   ├── BiRefNet
│   ├── pose_estimation
│   ├── SegmentAPI
├── example_data
│   └── test_data
├── exps
│   ├── releases
├── LHM
│   ├── datasets
│   ├── losses
│   ├── models
│   ├── outputs
│   ├── runners
│   ├── utils
│   ├── launch.py
├── pretrained_models
│   ├── dense_sample_points
│   ├── gagatracker
│   ├── human_model_files
│   ├── sam2
│   ├── sapiens
│   ├── voxel_grid
│   ├── arcface_resnet18.pth
│   ├── BiRefNet-general-epoch_244.pth
├── scripts
│   ├── exp
│   ├── convert_hf.py
│   └── upload_hub.py
├── tools
│   ├── metrics
├── train_data
│   ├── example_imgs
│   ├── motion_video
├── inference.sh
├── README.md
├── requirements.txt
```

###Qwen Image-Edit-2509 model download
```python
from huggingface_hub import snapshot_download

#Qwen-Imag-Edit-2509 (complete model, approximately 57GB)
model_dir = snapshot_download(
repo_id='Qwen/Qwen-Image-Edit-2509',
cache_dir='./pretrained_models/Qwen-Image-Edit-2509'
)
```

After downloading, the Qwen-Imag-Edit-2509 model file structure is as follows:
```bash
Qwen-Image-Edit-2509
├── processor
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── merges.txt
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── video_preprocessor_config.json
│   └── vocab.json
│
├── scheduler
│   └── scheduler_config.json
│
├── text_encoder
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00004.safetensors   # 4.97 GB
│   ├── model-00002-of-00004.safetensors   # 4.99 GB
│   ├── model-00003-of-00004.safetensors   # 4.93 GB
│   └── model-00004-of-00004.safetensors   # 1.69 GB
│
├── tokenizer
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
│
├── transformer
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── diffusion_pytorch_model-00001-of-00005.safetensors   # 9.97 GB
│   ├── diffusion_pytorch_model-00002-of-00005.safetensors   # 9.99 GB
│   ├── diffusion_pytorch_model-00003-of-00005.safetensors   # 9.99 GB
│   ├── diffusion_pytorch_model-00004-of-00005.safetensors   # 9.93 GB
│   └── diffusion_pytorch_model-00005-of-00005.safetensors   # 0.98 GB
│
├── vae
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors   # 254 MB
│
├── model_index.json
├── .gitattributes
└── README.md
```

##  📃  file path

>Please keep the path consistent with the following
>Omitting the content of the model itself
>The main working documents have been submitted by ✅ ️ Mark
```bash
Autodl-tmp
├── LHM                                  #  🟩  The main project designed for this lesson
│   ├── qwen_2509.py                     #  ✅ Qwen Local Call Script
│   ├── Qwen_API.py                      #  ✅ Qwen API version
│   ├── bucea.jpg                        # BUCEA Logo
│   ├── requirements.txt
│   ├── LICENSE
│   ├── README.md
│   ├── LHM # LHM Core Code
│   │   ├── datasets
│   │   ├── losses
│   │   ├── models
│   │   ├── runners
│   │   ├── utils
│   │   └── launch.py
│   │
│   ├── engine                     
│   │   ├── BiRefNet                     # If Sam is not used, use BiRefNet
│   │   ├── pose_estimation
│   │   └── SegmentAPI                   # SAM2 related
│   │
│   ├── pretrained-models                # All model weights
│   │   ├── Damo_XR_Lab
│   │   ├── 3DAIGC
│   │   ├── sam2
│   │   ├── gagatracker
│   │   ├── sapiens
│   │   ├── human_model_files
│   │   └── voxel_grid
│   │
│   ├── train_data                       # Input image/action video
│   │   ├── example_imgs
│   │   └── motion_video
│   │
│   ├── outputs
│   │   ├── ply # generated 3D Gaussian PLY
│   │   └── . ..
│   │
│   ├── configs
│   ├── scripts
│   └── tools
│
│
├── Qwen-Image-Edit-2509                 #  🟦  Local Qwen model (57GB)
│   ├── Qwen_inpaint.py                  #  ✅ Local image entry, called by LHM
│   ├── processor/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── transformer/
│   ├── vae/
│   └── model_index.json
│
│
├──  sam2 main/# SAM2 official warehouse (for segments)
├──  diffusers main/# HuggingFace diffusers full repository
├──  pytorch 3d main/# PyTorch 3D (LHM dependency)
├── BasicSR-master/
├── CLIP-main/
├── diff-gaussian-rasterization-main/
├── qwen_image_edit_api.py               #  ✅ Call API script, please enter your own API
└── tools/
```
##  💻  on-premises deployment
We now support users to call the Alibaba Qwen Image API to generate images. Users only need to write their own API key in the "qwen_image_edit_api. py" file.   
Application link: https://cn.aliyun.com/product/tongyi?from_alibabacloud=&utm_content=se_1021898286
```bash
#Call Qwen_ API to generate images
python ./Qwen_API.py 
#Call the local Qwen model to generate images
python ./Qwen_2509.py  
```

##  👀  Effect and Presentation
The following figure shows the operation of Qwen-API.Py
Images generated based on Qwen Image API and the results after Sam2 segmentation
  
<img src="./assets/QwenAPi.png" />

The following figure shows the operation of Qwen_2509-py
The image generated based on Qwen-Imag-Edit-2509 and the result after Sam2 segmentation
  
<img src="./assets/Qwen2509.png" />  
  
It can be seen that after using the diffusion model to preprocess the original input file, the effect has been significantly improved
The following figure shows a comparative experiment
  

<img src="./assets/ablation1.png" />  
  
It can be seen that the generation quality after adopting the Qwen model is the highest


The following figure shows the ablation experiment
  
<img src="./assets/ablation2.png" />  
  
It can be seen that using the Sam2+Qwem model for segmentation and preprocessing yields the best results


##  🤔  prospect
In fact, it can be found that the original intention of this task is to optimize the robustness of LHM in generating 3D digital humans
This experiment actually attempted to use the SDV-1.5 diffusion model, but the results were not satisfactory
With the introduction of Qwen Image-Edit-2509, the generation effect of the experiment was significantly improved, but at the same time, it also brought about shortcomings
① Generation time growth
② High usage of video memory
Future research could consider optimizing input images in a way that balances performance and cost


##  Thank you

This work is based on the following excellent research results and open-source projects:

- [LHM](https://github.com/aigc3d/LHM)
- [IDOL](https://yiyuzhuang.github.io/IDOL/)
