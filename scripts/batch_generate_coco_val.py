import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


@dataclass
class CocoPrompt:
    image_id: int
    file_name: str
    annotation_id: int
    caption: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate OmniGen2 images for COCO val2017 captions.")
    parser.add_argument(
        "--captions_json",
        type=str,
        default="/mnt/phwfile/datafrontier/zhangyue/mycode/coco/annotations/captions_val2017.json",
        help="Path to COCO captions JSON.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/mnt/phwfile/datafrontier/zhangyue/mycode/coco/val2017",
        help="Path to COCO val2017 images (groundtruth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="coco_val_gen_omnigen2/generated",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to OmniGen2 model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Optional transformer checkpoint path.",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Optional LoRA checkpoint path.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver++"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=5.0,
        help="Text guidance scale.",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale.",
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of CFG range.",
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of CFG range.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt text.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Limit number of images to generate (0 = all).",
    )
    parser.add_argument(
        "--captions_per_image",
        type=str,
        default="first",
        choices=["first", "all"],
        help="Use first caption or all captions per image.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if output file already exists.",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload.",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload.",
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload.",
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Enable teacache to speed up inference.",
    )
    parser.add_argument(
        "--teacache_rel_l1_thresh",
        type=float,
        default=0.05,
        help="Relative L1 threshold for teacache.",
    )
    parser.add_argument(
        "--enable_taylorseer",
        action="store_true",
        help="Enable TaylorSeer caching.",
    )
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if args.transformer_path:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    if args.transformer_lora_path:
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
        print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored.")

    if args.enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif args.enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    if args.scheduler == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(
            pipeline.transformer,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.mllm,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.vae,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
    else:
        pipeline = pipeline.to(accelerator.device)

    return pipeline


def load_coco_prompts(
    captions_json: str,
    images_dir: str,
    captions_per_image: str,
) -> List[CocoPrompt]:
    with open(captions_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images_by_id: Dict[int, str] = {
        image["id"]: image["file_name"]
        for image in payload.get("images", [])
    }

    prompts_by_image: Dict[int, List[CocoPrompt]] = {}
    for annotation in payload.get("annotations", []):
        image_id = annotation["image_id"]
        file_name = images_by_id.get(image_id)
        if not file_name:
            continue
        prompt = CocoPrompt(
            image_id=image_id,
            file_name=file_name,
            annotation_id=annotation["id"],
            caption=annotation["caption"],
        )
        prompts_by_image.setdefault(image_id, []).append(prompt)

    prompts: List[CocoPrompt] = []
    for image_id, items in prompts_by_image.items():
        if captions_per_image == "first":
            prompts.append(items[0])
        else:
            prompts.extend(items)

    prompts.sort(key=lambda item: (item.image_id, item.annotation_id))

    missing_images = [
        prompt.file_name for prompt in prompts
        if not os.path.exists(os.path.join(images_dir, prompt.file_name))
    ]
    if missing_images:
        print(f"WARNING: {len(missing_images)} groundtruth images missing under {images_dir}.")

    return prompts


def build_output_name(prompt: CocoPrompt, captions_per_image: str) -> str:
    stem, _ = os.path.splitext(prompt.file_name)
    if captions_per_image == "first":
        return f"{stem}.png"
    return f"{stem}_ann{prompt.annotation_id}.png"


def iter_prompts(prompts: List[CocoPrompt], max_items: int) -> Iterable[CocoPrompt]:
    if max_items <= 0:
        return prompts
    return prompts[:max_items]


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != "fp32" else "no")

    weight_dtype = torch.float32
    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = load_pipeline(args, accelerator, weight_dtype)
    prompts = load_coco_prompts(args.captions_json, args.images_dir, args.captions_per_image)

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")

    with open(metadata_path, "w", encoding="utf-8") as metadata_handle:
        for index, prompt in enumerate(iter_prompts(prompts, args.max_items)):
            output_name = build_output_name(prompt, args.captions_per_image)
            output_path = os.path.join(args.output_dir, output_name)
            if args.skip_existing and os.path.exists(output_path):
                continue

            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + index)
            result = pipeline(
                prompt=prompt.caption,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_step,
                max_sequence_length=1024,
                text_guidance_scale=args.text_guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                cfg_range=(args.cfg_range_start, args.cfg_range_end),
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
            )

            image = result.images[0]
            image.save(output_path)

            metadata = {
                "image_id": prompt.image_id,
                "annotation_id": prompt.annotation_id,
                "caption": prompt.caption,
                "groundtruth_path": os.path.join(args.images_dir, prompt.file_name),
                "generated_path": output_path,
            }
            metadata_handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
