# Copyright (c) 2023-2024, Qi Zuo & Lingteng Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

sys.path.append("/root/autodl-tmp")

import base64
import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image

torch._dynamo.config.disable = True
import argparse
import pdb
import shutil
import subprocess

from accelerate import Accelerator
from omegaconf import OmegaConf

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.model_download_utils import AutoModelQuery
from LHM.utils.model_query_utils import AutoModelSwitcher
from collections import defaultdict

try:
    from engine.SegmentAPI.SAM import SAM2Seg
except:
    print(
        "\033[31mNo SAM2 found! Try using rembg to remove the background. This may slightly degrade the quality of the results!\033[0m"
    )
    from rembg import remove

from engine.pose_estimation.video2motion import Video2MotionPipeline
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url, download_from_url
from LHM.utils.face_detector import VGGHeadDetector
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.gpu_utils import check_single_gpu_memory
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.model_card import MEMORY_MODEL_CARD, MODEL_CARD, MODEL_CONFIG
from LHM.utils.video_utils import get_video_hash


def qwen_image_edit(input_path, output_path, prompt=""):
    qwen_script = "/root/autodl-tmp/Qwen-Image-Edit-2509/Qwen_inpaint.py"

    if not os.path.exists(qwen_script):
        print(f"[Qwen ERROR] 找不到: {qwen_script}")
        return False

    # 使用 conda run —— 非交互式 subprocess 唯一不会报错的方法
    command = f"""
export CUDA_VISIBLE_DEVICES=1
/root/miniconda3/bin/conda run -n qwen_img python {qwen_script} \
    --input {input_path} \
    --output {output_path}
"""

    print("----------------------------------------------------")
    print("⚡ 本地 Qwen-2509 启动（conda run 模式）")
    print(f"➡ 输入: {input_path}")
    print(f"➡ 输出: {output_path}")
    print(f"➡ 环境: qwen_img")
    print(f"➡ GPU: 1")
    print("----------------------------------------------------")

    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("[Qwen STDOUT] --------------------------")
    print(result.stdout)
    print("[Qwen STDERR] --------------------------")
    print(result.stderr)

    if result.returncode != 0:
        print("[Qwen FAIL] 运行失败")
        return False

    print(f"[Qwen OK] 完成 → {output_path}")
    return True


def save_gaussians_to_ply(gs_model_list, output_path):
    import numpy as np

    gs = gs_model_list[0]

    print("====== LHM Gaussian Fields ======")
    print(dir(gs))
    print("==================================")

    # ---------- POSITION ----------
    if hasattr(gs, "offset_xyz"):
        xyz = gs.offset_xyz.detach().cpu().numpy()
    elif hasattr(gs, "xyz"):
        xyz = gs.xyz.detach().cpu().numpy()
    else:
        raise ValueError("No xyz or offset_xyz found in Gaussian model!")

    # ---------- COLOR ----------
    if not hasattr(gs, "shs"):
        raise ValueError("shs missing for color.")

    shs = gs.shs.detach().cpu().numpy()  # (N, C, SH_dim)
    N, C, SHdim = shs.shape

    print(f"[INFO] shs shape = {shs.shape}")

    dc = shs[:, :, 0]  # DC term (N, C)

    if C == 1:
        print("[WARN] SH color channel = 1, expanding to RGB.")
        dc = np.repeat(dc, 3, axis=1)
    elif C >= 3:
        dc = dc[:, :3]
    else:
        raise ValueError(f"Unsupported SH color channels: C={C}")

    rgb = np.clip(dc, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # ---------- OPACITY ----------
    if hasattr(gs, "opacity"):
        opac = gs.opacity.detach().cpu().numpy()  # (N,1)
    else:
        raise ValueError("opacity missing.")

    # ---------- SCALE ----------
    if hasattr(gs, "scaling"):
        scale = gs.scaling.detach().cpu().numpy()  # (N,3)
    else:
        raise ValueError("scaling missing.")

    # ---------- WRITE PLY ----------
    with open(output_path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float opacity\n"
            "property float scale_x\n"
            "property float scale_y\n"
            "property float scale_z\n"
            "end_header\n"
        )

        for i in range(N):
            f.write(
                f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} "
                f"{rgb_uint8[i, 0]} {rgb_uint8[i, 1]} {rgb_uint8[i, 2]} "
                f"{opac[i, 0]} "
                f"{scale[i, 0]} {scale[i, 1]} {scale[i, 2]}\n"
            )

    print(f"[✓] PLY saved → {output_path}")


def download_geo_files():
    if not os.path.exists("./pretrained_models/dense_sample_points/1_20000.ply"):
        download_from_url(
            "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply",
            "./pretrained_models/dense_sample_points/",
        )


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


def query_model_config(model_name):
    try:
        model_params = model_name.split("-")[1]

        return MODEL_CONFIG[model_params]
    except:
        return None


def prior_check():
    if not os.path.exists("./pretrained_models"):
        prior_data = MODEL_CARD["prior_model"]
        download_extract_tar_from_url(prior_data)


def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box


def infer_preprocess_image(
        rgb_path,
        mask,
        intr,
        pad_ratio,
        bg_color,
        max_tgt_size,
        aspect_standard,
        enlarge_ratio,
        render_tgt_size,
        multiply,
        need_mask=True,
):
    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1]: bbox_list[3], bbox_list[0]: bbox_list[2]]
    mask = mask[bbox_list[1]: bbox_list[3], bbox_list[0]: bbox_list[2]]

    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard

    target_w = int(min(w * scale_ratio, h))
    if target_w - w > 0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:
        from LHM.runners.infer.utils import scale_intrs

        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )
    return rgb, mask, intr


def parse_configs():
    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    query_model = AutoModelQuery()

    if os.environ.get("APP_MODEL_NAME") is not None:
        model_path = query_model.query(os.environ.get("APP_MODEL_NAME"))
        model_name = os.environ.get("APP_MODEL_NAME")
    else:
        raise NotImplementedError

    cli_cfg.model_name = model_path

    model_config = query_model_config(model_name)

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def _build_model(cfg):
    from LHM.models import model_dict

    hf_model_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    model = hf_model_cls.from_pretrained(cfg.model_name)

    return model


def animation_infer(renderer, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors):
    render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
        render_intrs[0, 0, 0, 2] * 2
    )
    render_res_list = []
    num_views = render_c2ws.shape[1]
    start_time = time.time()

    for view_idx in range(num_views):
        render_res = renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            renderer.get_single_view_smpl_data(smplx_params, view_idx),
            render_c2ws[:, view_idx: view_idx + 1],
            render_intrs[:, view_idx: view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx: view_idx + 1],
        )
        render_res_list.append(render_res)
    print(f"time elpased(animate gs model per frame):{(time.time() - start_time) / num_views}")

    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            if isinstance(v[0], torch.Tensor):
                out[k].append(v.detach().cpu())
            else:
                out[k].append(v)
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                out[k] = out[k][0].permute(0, 2, 3, 1)
        else:
            out[k] = v
    return out


def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")


def prepare_working_dir():
    working_dir = tempfile.TemporaryDirectory()
    return working_dir


def init_preprocessor():
    from LHM.utils.preprocess import Preprocessor

    global preprocessor
    preprocessor = Preprocessor()


def preprocess_fn(image_in: np.ndarray, remove_bg: bool, recenter: bool, working_dir):
    image_raw = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(image_in) as img:
        img.save(image_raw)
    image_out = os.path.join(working_dir.name, "rembg.png")
    success = preprocessor.preprocess(
        image_path=image_raw, save_path=image_out, rmbg=remove_bg, recenter=recenter
    )
    assert success, f"Failed under preprocess_fn!"
    return image_out


def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"


@torch.no_grad()
def demo_lhm(pose_estimator, face_detector, parsing_net, lhm, motion_generation, cfg):
    motion_processing_dir = "./train_data/users/motion_processing"
    if os.path.exists(motion_processing_dir):
        shutil.rmtree(motion_processing_dir)
    os.makedirs(motion_processing_dir, exist_ok=True)

    @spaces.GPU(duration=100)
    def core_fn(image: np.ndarray, video_params, working_dir):
        # --------------------------------------------------
        # 1. 保存原始输入图像
        # --------------------------------------------------
        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.fromarray(image) as img:
            img.save(image_raw)

        # --------------------------------------------------
        # 2. 无条件调用 Qwen 修复人物（不再判断身体完整性）
        # --------------------------------------------------
        print("[Qwen] Always performing full-body completion with Qwen...")

        sd_fixed_image_path = os.path.join(working_dir.name, "raw_qwen_fixed.png")

        try:
            success = qwen_image_edit(
                input_path=image_raw,
                output_path=sd_fixed_image_path,
                prompt="restore full body, extend to head-to-toe, complete limbs, photorealistic, same person"
            )
            if success:
                print("[Qwen] Completion success → using Qwen output")
                image_raw = sd_fixed_image_path
            else:
                print("[Qwen] Qwen returned failure → using original image")
        except Exception as e:
            print("[Qwen ERROR]", e)
            print("[Qwen] Using original image.")

        # --------------------------------------------------
        # 3. 修复后再跑 SAM2 segmentation
        # --------------------------------------------------
        mask_to_use = None
        if parsing_net is not None:
            try:
                parsing_out = parsing_net(img_path=image_raw, bbox=None)
                mask_to_use = (parsing_out.masks * 255).astype(np.uint8)
                print("[SAM2] Segmentation after Qwen fix done.")
            except Exception as e:
                print("[SAM2 ERROR]", e)
                mask_to_use = None

        # SAM2 失败兜底：用全 1 mask（整张图当做人）
        if mask_to_use is None:
            rgb_tmp = np.array(Image.open(image_raw))
            h, w = rgb_tmp.shape[:2]
            mask_to_use = np.ones((h, w), dtype=np.uint8) * 255

        # --------------------------------------------------
        # 4. Motion 相关路径 & 配置（与原版 LHM 一致）
        # --------------------------------------------------
        base_vid = os.path.basename(video_params).split(".")[0]
        smplx_params_dir = os.path.join(
            "./train_data/motion_video/", base_vid, "smplx_params"
        )

        if not os.path.exists(smplx_params_dir):
            motion_processing_dir = "./train_data/users/motion_processing"
            video_hash = get_video_hash(video_params)
            output_path = os.path.join(motion_processing_dir, video_hash)

            if not os.path.exists(output_path):
                smplx_params_dir = motion_generation(
                    video_params, output_path, is_file_only=True
                )
            else:
                smplx_params_dir = os.path.join(output_path, "smplx_params")

        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_image_path = os.path.join(working_dir.name, "output.png")

        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.rsplit(".", 1)[0]  # 只去掉扩展名

        subdir_path = os.path.dirname(image_raw).replace(omit_prefix, "")
        subdir_path = subdir_path[1:] if subdir_path.startswith("/") else subdir_path
        print("subdir_path and uid:", subdir_path, uid)

        motion_seqs_dir = smplx_params_dir

        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )
        motion_name = os.path.basename(motion_name)

        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)

        print(image_raw, motion_seqs_dir, dump_image_dir, dump_video_path)

        dump_tmp_dir = dump_image_dir

        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30

        aspect_standard = 5.0 / 3
        motion_img_need_mask = cfg.get("motion_img_need_mask", False)
        vis_motion = cfg.get("vis_motion", False)

        # --------------------------------------------------
        # 5.（保留 PoseEstimator，只用来取 beta，不做 full-body 判断）
        # --------------------------------------------------
        with torch.no_grad():
            shape_pose = pose_estimator(image_raw)  # 不再检查 is_full_body

        # --------------------------------------------------
        # 6. 图像预处理（使用 Qwen+SAM2 后的 mask）
        # --------------------------------------------------
        image, _, _ = infer_preprocess_image(
            image_raw,
            mask=mask_to_use,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        # --------------------------------------------------
        # 7. 头部裁剪 + resize
        # --------------------------------------------------
        try:
            rgb = np.array(Image.open(image_raw))[..., :3]
            rgb = torch.from_numpy(rgb).permute(2, 0, 1)
            bbox = face_detector.detect_face(rgb)
            head_rgb = rgb[
                       :, int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])
                       ]
            head_rgb = head_rgb.permute(1, 2, 0)
            src_head_rgb = head_rgb.cpu().numpy()
        except Exception:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)

        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(cfg.src_head_size, cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            src_head_rgb = np.zeros(
                (cfg.src_head_size, cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        save_ref_img_path = os.path.join(dump_tmp_dir, "output.png")
        vis_ref_img = (
                image[0].permute(1, 2, 0).cpu().detach().numpy() * 255
        ).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )
        motion_name = os.path.basename(motion_name)

        # --------------------------------------------------
        # 8. 准备 motion 序列
        # --------------------------------------------------
        motion_seq = prepare_motion_seqs(
            motion_seqs_dir,
            None,
            save_root=dump_tmp_dir,
            fps=30,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
            motion_size=3000,
        )

        camera_size = len(motion_seq["motion_seqs"])
        shape_param = shape_pose.beta

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        lhm.to(dtype)

        smplx_params = motion_seq["smplx_params"]
        smplx_params["betas"] = shape_param.to(device)

        # --------------------------------------------------
        # 9. 单视角重建 GS
        # --------------------------------------------------
        gs_model_list, query_points, transform_mat_neutral_pose = lhm.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        )

        ply_dir = "./outputs/ply"
        os.makedirs(ply_dir, exist_ok=True)
        ply_path = os.path.join(ply_dir, f"{uid}_gaussians.ply")

        print(f"[LHM] Saving Gaussian PLY to {ply_path} ...")
        save_gaussians_to_ply(gs_model_list, ply_path)

        # --------------------------------------------------
        # 10. 动画渲染
        # --------------------------------------------------
        start_time = time.time()
        batch_list = []
        batch_size = 40

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                print(f"batch: {batch_i}, total: {camera_size // batch_size + 1} ")

                keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]

                batch_smplx_params = dict()
                batch_smplx_params["betas"] = shape_param.to(device)
                batch_smplx_params["transform_mat_neutral_pose"] = (
                    transform_mat_neutral_pose
                )
                for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                                              :, batch_i: batch_i + batch_size
                                              ].to(device)

                res = lhm.animation_infer(
                    gs_model_list,
                    query_points,
                    batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                                :, batch_i: batch_i + batch_size
                                ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                                 :, batch_i: batch_i + batch_size
                                 ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                                     :, batch_i: batch_i + batch_size
                                     ].to(device),
                )

                if batch_i == 0:
                    gs_raw = res["3dgs"]
                    while isinstance(gs_raw, (list, tuple)):
                        gs_raw = gs_raw[0]

                    deformed_gs = gs_raw

                    export_dir = "./outputs/ply"
                    os.makedirs(export_dir, exist_ok=True)
                    export_path = os.path.join(export_dir, f"{uid}_frame0.ply")

                    print(f"[LHM] Exporting deformed frame Gaussian → {export_path}")
                    save_gaussians_to_ply([deformed_gs], export_path)

            comp_rgb = res["comp_rgb"]
            comp_mask = res["comp_mask"]
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (
                (batch_rgb.clamp(0, 1) * 255)
                .to(torch.uint8)
                .detach()
                .cpu()
                .numpy()
            )
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()

        rgb = np.concatenate(batch_list, axis=0)
        print(f"time elapsed: {time.time() - start_time}")

        if vis_motion:
            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]))[
                None, :, :, :
                ],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate(
                [rgb, motion_seq["vis_motion_render"], vis_ref_img], axis=2
            )

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

        # 返回：裁剪后的参考图 / 渲染视频 / Qwen 修复后的图
        return dump_image_path, dump_video_path, sd_fixed_image_path

    _TITLE = """LHM: Large Animatable Human Model"""

    _DESCRIPTION = """
        <strong>Reconstruct a human avatar in 0.2 seconds with A100!</strong>
    """

    with gr.Blocks(analytics_enabled=False) as demo:
        # ---------------------------- 自定义标题 + logo ----------------------------
        logo_url = "/root/autodl-tmp/LHM/bucea.jpg"
        logo_base64 = get_image_base64(logo_url)

        # 🔥 统一自适应样式：canvas + video + examples
        gr.HTML("""
        <style>
        /* 所有图片 canvas 自适应 */
        #input-img canvas,
        #processed-img canvas,
        #sd-fixed-img canvas {
            width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
        }

        /* video 自适应 */
        .auto-resize video {
            width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
        }

        /* Gradio 示例图片 */
        .gradio-examples img {
            width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
        }
        </style>
        """)

        gr.HTML(
            f"""
            <div style="text-align:center">
                <h1>
                    <img src="{logo_base64}" style='height:45px;vertical-align:middle;border-radius:6px;margin-right:10px;'/>
                    智研251 廖偲宇 · 基于LHM与Qwen-image的三维高清数字人生成
                </h1>
            </div>
            """
        )

        # ------------------------ 第一行（输入图 + 预处理图） ------------------------
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    value="./train_data/example_imgs/00000000_joker_2.jpg",
                    image_mode="RGBA",
                    type="numpy",
                    sources="upload",
                    elem_id="input-img",  # ★ 关键：用于匹配 CSS
                )

                # 示例图片
                if os.path.exists("./train_data/example_imgs/"):
                    example_imgs = [
                        os.path.join("./train_data/example_imgs/", f)
                        for f in os.listdir("./train_data/example_imgs/")
                    ]
                    gr.Examples(
                        examples=example_imgs,
                        inputs=[input_image],
                        examples_per_page=6,
                    )

            with gr.Column(scale=1):
                processed_image = gr.Image(
                    label="Processed (Cropped) Image",
                    image_mode="RGBA",
                    type="filepath",
                    interactive=False,
                    elem_id="processed-img",  # ★ 关键
                )

        # ------------------------ 第二行（Qwen 修复图） ------------------------
        sd_fixed_image = gr.Image(
            label="SD Fixed Image (after diffusion completion)",
            image_mode="RGBA",
            type="filepath",
            elem_id="sd-fixed-img",  # ★ 关键
        )

        # ------------------------ 第三行（动作视频 + 渲染结果） ------------------------
        with gr.Row():
            with gr.Column(scale=1):
                # 准备 motion 示例
                default_video = None
                new_examples = []
                if os.path.exists("./train_data/motion_video/"):
                    examples_video = os.listdir("./train_data/motion_video/")
                    examples = [
                        os.path.join(
                            "./train_data/motion_video/", example, "samurai_visualize.mp4"
                        )
                        for example in examples_video
                    ]
                    examples = sorted(examples)
                    for example in examples:
                        video_basename = os.path.basename(os.path.dirname(example))
                        input_video_path = os.path.join(
                            os.path.dirname(example), video_basename + ".mp4"
                        )
                        if not os.path.exists(input_video_path):
                            shutil.copyfile(example, input_video_path)
                        new_examples.append(input_video_path)
                    if len(new_examples) > 0:
                        default_video = new_examples[0]

                video_input = gr.Video(
                    label="Target Motion",
                    sources="upload",
                    elem_classes="auto-resize",  # 视频自适应
                    value=default_video,
                )

                if new_examples:
                    gr.Examples(
                        examples=new_examples,
                        inputs=[video_input],
                        examples_per_page=6,
                    )

            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Rendered Video",
                    format="mp4",
                    elem_classes="auto-resize",  # 视频自适应
                    autoplay=True,
                )

        # ------------------------ 按钮 ------------------------
        submit = gr.Button("Generate", variant="primary")
        working_dir = gr.State()

        submit.click(
            fn=assert_input_image,
            inputs=[input_image],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=core_fn,
            inputs=[input_image, video_input, working_dir],
            outputs=[processed_image, output_video, sd_fixed_image],
        )

        demo.queue()
        demo.launch(server_name="0.0.0.0")


def get_parse():
    parser = argparse.ArgumentParser(
        description="LHM-gradio: Large Animatable Human Model"
    )
    parser.add_argument(
        "--model_name",
        default="LHM-1B-HF",
        type=str,
        choices=["LHM-500M", "LHM-1B", "LHM-500M-HF", "LHM-1B-HF", "LHM-MINI"],
        help="Model name",
    )
    args = parser.parse_args()
    return args


def launch_gradio_app():
    args = get_parse()

    model_name = args.model_name
    model_switcher = AutoModelSwitcher(MEMORY_MODEL_CARD, extra_memory=6000)
    model_name = model_switcher.query(model_name)

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": model_name,
            "APP_TYPE": "infer.human_lrm",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    prior_check()
    download_geo_files()

    device = avaliable_device()

    motion_generation = Video2MotionPipeline(
        "./pretrained_models/human_model_files",
        fitting_steps=[30, 50],
        device=device,
        kp_mode="vitpose",
        visualize=False,
        pad_ratio=0.2,
        fov=60,
    )

    facedetector = VGGHeadDetector(
        "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
        device=device,
    )
    facedetector.to(device)

    pose_estimator = PoseEstimator(
        "./pretrained_models/human_model_files/", device="cpu"
    )
    pose_estimator.to(device)
    pose_estimator.device = device
    try:
        parsingnet = SAM2Seg()
    except Exception:
        parsingnet = None

    accelerator = Accelerator()

    cfg, cfg_train = parse_configs()
    lhm = _build_model(cfg)
    lhm.to("cuda")

    demo_lhm(pose_estimator, facedetector, parsingnet, lhm, motion_generation, cfg)


if __name__ == "__main__":
    launch_gradio_app()
