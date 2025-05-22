from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLAdapterPipeline,
    AutoencoderKL,
    UniPCMultistepScheduler,
    T2IAdapter,
)
import torch, os
from PIL import Image
from io import BytesIO
import models
from database import SessionLocal
from text_processor import (
    get_resolved_sentences,
    detect_and_translate_to_english,
    get_script_captions,
)
from s3 import upload_image_to_s3
from diffusers.utils import load_image
import random
from controlnet_aux import OpenposeDetector
import numpy as np
import gc

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# Initialize global generator
generator = torch.Generator()

# Initialize the models globally to ensure they're only loaded once
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype, use_safetensors=True
).to(device)

print("Loading base pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True,
    attn_implementation="xformers",
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Load LoRA weights
pipe.load_lora_weights(
    "safetensors/Storyboard_sketch.safetensors", adapter_name="sketch"
)
pipe.load_lora_weights("safetensors/anglesv2.safetensors", adapter_name="angles")
pipe.set_adapters(["sketch", "angles"], adapter_weights=[0.5, 0.5])
pipe.enable_xformers_memory_efficient_attention()

print("Loading OpenPose detector...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

print("Loading T2I adapter...")
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-openpose-sdxl-1.0", torch_dtype=dtype
)

print("Loading adapter pipeline...")
posepipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    adapter=adapter,
    vae=vae,
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True,
    attn_implementation="xformers",
)

posepipe.enable_model_cpu_offload()
posepipe.scheduler = UniPCMultistepScheduler.from_config(posepipe.scheduler.config)

posepipe.load_lora_weights(
    "safetensors/Storyboard_sketch.safetensors", adapter_name="sketch"
)
posepipe.load_lora_weights("safetensors/anglesv2.safetensors", adapter_name="angles")
posepipe.set_adapters(["sketch", "angles"], adapter_weights=[0.5, 0.5])
posepipe.enable_xformers_memory_efficient_attention()

print("All models loaded successfully")


def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_dimensions(resolution: str) -> tuple[int, int]:
    resolution_map = {
        "16:9": (1024, 576),
        "1:1": (1024, 1024),
        "9:16": (576, 1024),
    }
    return resolution_map.get(resolution, (1024, 1024))


def generate_batch_images(
    story: str, storyboard_id: int, resolution: str = "1:1", isStory: bool = True
):
    # Clear cache before batch generation
    clear_cuda_cache()

    db = SessionLocal()
    try:
        if isStory:
            prompts = get_resolved_sentences(story)
        elif not isStory:
            prompts = get_script_captions(story)

        width, height = get_dimensions(resolution)

        for num, prompt in enumerate(prompts):
            # Generate a random seed for each image in the batch
            seed = random.randint(0, 2**32 - 1)
            generator.manual_seed(seed)

            print(f"Generating image {num+1} with seed {seed}")

            result = pipe(
                prompt=f"Storyboard sketch of {prompt}, black and white, cinematic, high quality",
                negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, abstract, bad physics",
                guidance_scale=8.5,
                height=height,
                width=width,
                num_inference_steps=30,
                generator=generator,
            )

            image = result.images[0]
            buf = BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)

            s3_url = upload_image_to_s3(
                buf.read(),
                f"image_{num + 1}.jpg",
                folder=f"storyboards/{storyboard_id}",
            )

            db_image = models.Image(
                storyboard_id=storyboard_id,
                image_path=s3_url,
                caption=prompt,
            )
            db.add(db_image)
            db.commit()
            db.refresh(db_image)

            print(f"Image {num+1} generated successfully")

            # Clear cache after each image
            clear_cuda_cache()

    except Exception as e:
        print(f"Error during image generation: {e}")
        import traceback

        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


def generate_single_image(
    image_id: int,
    caption: str,
    seed: int = None,
    resolution: str = "1:1",
    isOpenPose: bool = False,
    pose_img: Image.Image = None,
):
    # Clear cache before single image generation
    clear_cuda_cache()

    db = SessionLocal()
    try:
        # Get existing image record
        db_image = db.query(models.Image).filter(models.Image.id == image_id).first()
        processed_caption = detect_and_translate_to_english(caption)
        width, height = get_dimensions(resolution)

        # Use provided seed or generate a random one
        current_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        generator.manual_seed(current_seed)

        print(f"Generating single image with seed {current_seed}")

        if not db_image:
            raise ValueError(f"Image with id {image_id} not found.")

        if isOpenPose:
            print("Using OpenPose pipeline")
            image = openpose(pose_img, detect_resolution=512, image_resolution=1024)
            image = np.array(image)[:, :, ::-1]
            image = Image.fromarray(np.uint8(image))

            result = posepipe(
                prompt=f"Storyboard sketch of {processed_caption}, black and white, cinematic, high quality",
                negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, abstract, bad physics",
                image=image,
                adapter_conditioning_scale=1,
                guidance_scale=8.5,
                num_inference_steps=30,
                generator=generator,
            )
        else:
            print("Using standard pipeline")
            result = pipe(
                prompt=f"Storyboard sketch of {processed_caption}, black and white, cinematic, high quality",
                negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, abstract, bad physics",
                guidance_scale=8.5,
                num_inference_steps=30,
                width=width,
                height=height,
                generator=generator,
            )

        # Save and upload
        image = result.images[0]
        buf = BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        s3_url = upload_image_to_s3(
            buf.read(),
            f"image_{image_id}.jpg",
            folder=f"storyboards/{db_image.storyboard_id}",
        )

        # Update image record
        db_image.image_path = s3_url
        db_image.caption = caption
        db_image.seed = current_seed
        db.commit()
        db.refresh(db_image)

        print(f"Single image generated successfully")

        # Clear cache after generation
        clear_cuda_cache()

        return db_image

    except Exception as e:
        print(f"Error during image regeneration: {e}")
        import traceback

        traceback.print_exc()
        db.rollback()
        return None
    finally:
        db.close()
