import os
import time
import shutil
import json
import base64
import random
import threading
import asyncio
import logging
from datetime import datetime, timezone

import uuid
from typing import Dict

import gradio as gr
import numpy as np
import torch
import spaces
from PIL import Image

# ---- Diffusers / Transformers ----
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel

# =========================
# Logging
# =========================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# Prompt Polisher
# =========================
SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the edited input image’s overall scene.

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.).
- Remove meaningless instructions (e.g., "Add 0 objects").
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.
- For text replacement tasks, always use:
    - `Replace "xx" to "yy"`.
    - `Replace the xx bounding box to "yy"`.
- If no text is given, infer a concise text consistent with the image context.
- Specify text position, color, and layout concisely.

### 3. Human Editing Tasks
- Maintain the person’s core visual consistency.
- Changes (e.g., clothes, hairstyle) must match the original style.
- Expression changes must be subtle and natural.
- Preserve the main subject unless deletion is explicit.

### 4. Style / Enhancement
- Describe styles concisely using key traits.
- For “keep current style”, extract dominant features and integrate.
- For photo restoration, use:
  "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"

## 3. Logic Checks
- Resolve contradictions.
- Fill missing key info (e.g., reasonable position).
'''

async def ping():
    await asyncio.sleep(0)
    payload = {
        "status": "running",
        "message": "pong",
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.debug("PING %s", payload["time_utc"])
    return payload

def encode_image(pil_image: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def api(prompt, img_list, model="qwen-vl-max-latest", kwargs={}):
    import dashscope
    api_key = os.environ.get('DASH_API_KEY')
    if not api_key:
        logger.error("DASH_API_KEY is not set")
        raise EnvironmentError("DASH_API_KEY is not set")
    assert model in ["qwen-vl-max-latest"], f"Not implemented model {model}"

    sys_promot = "you are a helpful assistant, you should provide useful answers to users."
    messages = [
        {"role": "system", "content": sys_promot},
        {"role": "user", "content": []},
    ]
    for img in img_list:
        messages[1]["content"].append({"image": f"data:image/png;base64,{encode_image(img)}"})
    messages[1]["content"].append({"text": f"{prompt}"})

    response_format = kwargs.get('response_format', None)

    t0 = time.perf_counter()
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format='message',
            response_format=response_format,
        )
    except Exception:
        logger.exception("DashScope call failed in %.3fs", time.perf_counter() - t0)
        raise

    dt = time.perf_counter() - t0
    if response.status_code == 200:
        logger.info("DashScope ok model=%s duration=%.3fs", model, dt)
        return response.output.choices[0].message.content[0]['text']
    else:
        logger.error("DashScope error status=%s duration=%.3fs body=%s",
                     response.status_code, dt, getattr(response, "output_text", ""))
        raise Exception(f'Failed to post: {response}')

def polish_prompt(prompt, img):
    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    attempts, t0 = 0, time.perf_counter()
    while True:
        attempts += 1
        try:
            result = api(prompt, [img])
            if isinstance(result, str):
                result = result.replace('```json', '').replace('```', '')
                result = json.loads(result)
            polished_prompt = result['Rewritten'].strip().replace("\n", " ")
            logger.info("polish_prompt success attempts=%d duration=%.3fs",
                        attempts, time.perf_counter() - t0)
            return polished_prompt
        except Exception as e:
            logger.warning("polish_prompt attempt=%d error=%s", attempts, e)

# =========================
# Globals / Concurrency
# =========================
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max

PIPE_LOCK = threading.Lock()          # protects adapter mutations/calls
_FAST_GATE = threading.Semaphore(1)   # infer_fast: max 1 concurrent

# One-time init gates (process-wide)
_INIT_LOCK = threading.Lock()
_INIT_READY = threading.Event()
_FAST = {"pipe": None, "model_id": None}

_FULL_INIT_LOCK = threading.Lock()
_FULL_READY = threading.Event()
_FULL_PIPE = {"pipe": None}

enableFigure = False

# =========================
# Thread-safe one-time init
# =========================
def init_models():
    """
    Thread-safe, one-time init for the FAST pipeline.
    First caller initializes; concurrent callers wait until ready.
    """
    if _INIT_READY.is_set():
        return _FAST

    if _INIT_LOCK.acquire(blocking=False):
        try:
            if _INIT_READY.is_set():
                return _FAST

            logger.info("Initializing models (one-time, fast pipe only)")
            model_id = "Qwen/Qwen-Image-Edit"

            # Quantized transformer (Diffusers) on CPU to avoid CUDA warmup
            quant_config_diff = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            )
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quant_config_diff,
                torch_dtype=torch.bfloat16,
            ).to("cpu")

            # Quantized text encoder (Transformers) on CPU
            quant_config_tx = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                subfolder="text_encoder",
                quantization_config=quant_config_tx,
                torch_dtype=torch.bfloat16,
            ).to("cpu")

            # Fast edit pipeline (CPU offload will migrate as needed)
            pipe = QwenImageEditPipeline.from_pretrained(
                model_id,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=torch.bfloat16
            )

            # Load LoRAs once
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
                adapter_name="lightning"
            )
            pipe.load_lora_weights(
                "/root/Qwen-Image/src/examples/ootd_colour-19-3600.safetensors",
                adapter_name="realism"
           )

            # Default mix (can tweak per-request under PIPE_LOCK)
            pipe.set_adapters(["lightning", "realism"], adapter_weights=[1, 0.65])

            # One-time: enable cpu offload to control VRAM
            pipe.enable_model_cpu_offload()

            _FAST["pipe"] = pipe
            _FAST["model_id"] = model_id

            _INIT_READY.set()
            logger.info("Fast pipe ready")
            return _FAST
        finally:
            _INIT_LOCK.release()
    else:
        _INIT_READY.wait()
        return _FAST

def init_full():
    """
    Thread-safe, one-time init for the FULL pipeline (lazy, on-demand).
    """
    if _FULL_READY.is_set():
        return _FULL_PIPE["pipe"]

    if _FULL_INIT_LOCK.acquire(blocking=False):
        try:
            if _FULL_READY.is_set():
                return _FULL_PIPE["pipe"]

            logger.info("Initializing full pipeline (one-time, on demand)")
            model_id = _FAST["model_id"] or "Qwen/Qwen-Image-Edit"

            pipe_full = QwenImageEditPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
            if device == "cuda":
                try:
                    pipe_full = pipe_full.to(device)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM moving full pipe to CUDA; enabling CPU offload instead")
                    pipe_full.enable_model_cpu_offload()
            _FULL_PIPE["pipe"] = pipe_full
            _FULL_READY.set()
            logger.info("Full pipe ready")
            return pipe_full
        finally:
            _FULL_INIT_LOCK.release()
    else:
        _FULL_READY.wait()
        return _FULL_PIPE["pipe"]

# =========================
# FULL generation
# =========================
def infer_full(
    image: Image.Image,
    prompt: str,
    seed: int = 42,
    randomize_seed: bool = False,
    true_guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    width: int = 1024,
    height: int = 1024,
    ootd: float = 1.0,
    figure: float = 0.0,
    rewrite_prompt: bool = False,
    num_images_per_prompt: int = 1,
    progress=gr.Progress(track_tqdm=True),
):
    t0 = time.perf_counter()
    negative_prompt = " "

    # Ensure fast is ready (optional) then get full
    init_models()
    pipe_full = init_full()

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)

    if rewrite_prompt:
        prompt = polish_prompt(prompt, image)
        logger.debug("[Full] Rewritten Prompt: %s", prompt)

    logger.info("[Full] start seed=%s steps=%s guidance=%.3f size=%dx%d",
                seed, num_inference_steps, true_guidance_scale, width, height)
    images = pipe_full(
        image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt
    ).images
    logger.info("[Full] done duration=%.3fs", time.perf_counter() - t0)
    return images, seed

TMP_DIR = "/tmp/gradio"
LAST_TIME = 0

def _cleanup_old_files(max_age_sec: int = 10800) -> None:
    """
    Delete files whose recorded creation time is older than max_age_sec.
    Missing files are silently ignored. Stale registry entries are pruned.
    """
    global LAST_TIME
    now = time.time()
    if now - LAST_TIME > max_age_sec:
        try:
            time.sleep(5)
            LAST_TIME = time.time()
            if os.path.exists(TMP_DIR):
                shutil.rmtree(TMP_DIR)
                print(f"Deleted: {TMP_DIR}")
            else:
                print(f"Folder not found: {TMP_DIR}")
        except Exception:
                # Best-effort cleanup: continue even if one file fails
            pass

# =========================
# FAST generation (single-concurrency)
# =========================
def infer_fast(
    image: Image.Image,
    prompt: str,
    seed: int = 42,
    randomize_seed: bool = False,
    true_guidance_scale: float = 1.0,   # kept for symmetry; ignored
    num_inference_steps: int = 50,      # clamped to 8 (Lightning)
    width: int = 1024,
    height: int = 1024,
    ootd: float = 1.0,
    figure: float = 0.0,
    rewrite_prompt: bool = False,
    num_images_per_prompt: int = 1,
    progress=gr.Progress(track_tqdm=True),
):
    # Guarantee max 1 concurrent fast job
    if not _FAST_GATE.acquire(blocking=False):
        logger.warning("[Fast] rejected: another fast job is running")
        raise gr.Error("Another fast generation is running. Please try again in a moment.")
    t0 = time.perf_counter()
    try:
        global enableFigure
        os.makedirs(TMP_DIR, exist_ok=True)
        _cleanup_old_files()
        # Thread-safe, one-time init; wait if another request is initializing
        M = init_models()
        pipe = M["pipe"]
        os.makedirs(TMP_DIR, exist_ok=True)

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        dvc = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=dvc).manual_seed(seed)

        if rewrite_prompt:
            prompt = polish_prompt(prompt, image)
            logger.debug("[Fast] Rewritten Prompt: %s", prompt)

        steps = min(num_inference_steps, 8)
        logger.info("[Fast] start seed=%s steps=%s size=%dx%d ootd=%.2f",
                    seed, steps, width, height, ootd)

        # If you tweak adapters per request, guard mutation:
        with PIPE_LOCK:
            if(figure == 0.0 and enableFigure == True):
                enableFigure = False
                pipe.unload_lora_weights()
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Lightning",
                     weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
                     adapter_name="lightning"
                )
                pipe.load_lora_weights(
                    "/root/Qwen-Image/src/examples/ootd_colour-19-3600.safetensors",
                    adapter_name="realism"
                )
            if(figure == 1.0 and enableFigure == False):
                enableFigure = True
                pipe.unload_lora_weights()
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Lightning",
                     weight_name="Qwen-Image-Lightning-8steps-V2.0.safetensors",
                     adapter_name="lightning"
                )
                pipe.load_lora_weights(
                     "/root/Qwen-Image/src/examples/aldniki_qwen_figure_maker_v01.safetensors",
                     adapter_name="figure"
                )
            if(enableFigure == True):
                pipe.set_adapters(["lightning", "figure"], adapter_weights=[1, figure])
            else:
                pipe.set_adapters(["lightning", "realism"], adapter_weights=[1, ootd])
            out = pipe(
                image,
                prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                generator=generator
            ).images

        logger.info("[Fast] done duration=%.3fs", time.perf_counter() - t0)
        return out, seed
    except Exception:
        logger.exception("[Fast] failed duration=%.3fs", time.perf_counter() - t0)
        raise
    finally:
        _FAST_GATE.release()

# =========================
# Dispatcher
# =========================
def infer_dispatch(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    width=1024,
    height=1024,
    ootd=1.0,
    figure=0.0,
    rewrite_prompt=False,
    num_images_per_prompt=1,
    fast=True,
    progress=gr.Progress(track_tqdm=True),
):
    logger.debug("dispatch fast=%s seed=%s steps=%s size=%dx%d",
                 fast, seed, num_inference_steps, width, height)
    if fast:
        return infer_fast(
            image, prompt, seed, randomize_seed, true_guidance_scale,
            num_inference_steps, width, height, ootd, figure, rewrite_prompt, num_images_per_prompt
        )
    else:
        return infer_full(
            image, prompt, seed, randomize_seed, true_guidance_scale,
            num_inference_steps, width, height, ootd, figure, rewrite_prompt, num_images_per_prompt
        )

# =========================
# UI
# =========================
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown("[Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit) to run locally with ComfyUI or diffusers.")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", show_label=False, type="pil")
            result = gr.Gallery(label="Result", show_label=False, type="pil")

        with gr.Row():
            prompt_in = gr.Text(
                label="Prompt",
                show_label=False,
                placeholder="describe the edit instruction",
                container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed_in = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed_in = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale_in = gr.Slider(label="True guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=4.0)
                num_inference_steps_in = gr.Slider(label="Number of inference steps", minimum=1, maximum=24, step=1, value=8)
                width = gr.Slider(label="Width", minimum=256, maximum=4096, step=1, value=1024)
                height = gr.Slider(label="Height", minimum=256, maximum=4096, step=1, value=1024)
                ootd = gr.Slider(label="OOTD", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                figure = gr.Slider(label="Figure", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        rewrite_prompt_state = gr.State(False)
        num_images_per_prompt_state = gr.State(1)
        fast_state = gr.State(True)

        api_only_fast = gr.Button(visible=False)
        api_only_full = gr.Button(visible=False)

    gr.on(
        triggers=[run_button.click, prompt_in.submit],
        fn=infer_dispatch,
        inputs=[
            input_image,
            prompt_in,
            seed_in,
            randomize_seed_in,
            true_guidance_scale_in,
            num_inference_steps_in,
            width,
            height,
            ootd,
            figure,
            rewrite_prompt_state,
            num_images_per_prompt_state,
            fast_state,
        ],
        outputs=[result, seed_in],
        api_name="generate"
    )

    api_only_fast.click(
        fn=infer_fast,
        inputs=[
            input_image,
            prompt_in,
            seed_in,
            randomize_seed_in,
            true_guidance_scale_in,
            num_inference_steps_in,
            width,
            height,
            ootd,
            figure,
            rewrite_prompt_state,
            num_images_per_prompt_state,
        ],
        outputs=[result, seed_in],
        api_name="generate_fast"
    )

    out = gr.JSON(visible=False)
    demo.load(fn=ping, inputs=None, outputs=out, api_name="ping", queue=False)

    api_only_full.click(
        fn=infer_full,
        inputs=[
            input_image,
            prompt_in,
            seed_in,
            randomize_seed_in,
            true_guidance_scale_in,
            num_inference_steps_in,
            width,
            height,
            ootd,
            figure,
            rewrite_prompt_state,
            num_images_per_prompt_state,
        ],
        outputs=[result, seed_in],
        api_name="generate_full"
    )

if __name__ == "__main__":
    # Optional warmup so first request doesn't pay init cost
    try:
        init_models()
    except Exception:
        logger.exception("Warmup failed (continuing without warmup)")
    logger.info("Launching Gradio app on 0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=21450)
