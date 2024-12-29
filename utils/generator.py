import torch
import inspect
import logging
import torchvision.transforms as transforms
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


module = inspect.getmodule(StableDiffusionSafetyChecker)
logging.getLogger(module.__name__).setLevel(logging.ERROR)


class Text2ImageWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.server_generator == 'StableDiffusion':
            self.GenPipe = AutoPipelineForText2Image.from_pretrained(
                'generator/stable-diffusion-v1-5', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                local_files_only=True, 
                use_safetensors=True,
            )
        elif args.server_generator == 'StableDiffusionXL':
            self.GenPipe = AutoPipelineForText2Image.from_pretrained(
                'generator/stable-diffusion-xl-base-1.0', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                local_files_only=True, 
                use_safetensors=True,
            )
        elif args.server_generator == 'OpenJourney':
            self.GenPipe = AutoPipelineForText2Image.from_pretrained(
                'generator/openjourney', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                local_files_only=True, 
                use_safetensors=True,
            )
        elif args.server_generator == 'FLUX':
            self.GenPipe = AutoPipelineForText2Image.from_pretrained(
                'generator/FLUX.1-dev', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                max_memory={0:'10GB'}, 
                local_files_only=True, 
                use_safetensors=True,
            )
        else:
            raise NotImplementedError
        
        self.GenPipe.set_progress_bar_config(disable=True)
        self.img_size = max(512, args.img_size)


    def __call__(self, prompt, img, negative_prompt):
        with torch.no_grad():
            if self.args.server_generator == 'FLUX':
                res = self.GenPipe(prompt=prompt, 
                    height=self.img_size, 
                    width=self.img_size, 
                    num_images_per_prompt=self.args.num_images_per_prompt, 
                )
            else:
                res = self.GenPipe(prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    height=self.img_size, 
                    width=self.img_size, 
                    num_images_per_prompt=self.args.num_images_per_prompt, 
                )

            if self.args.server_generator in ['StableDiffusionXL', 'FLUX']:
                return res.images
            else:
                generated_images = []
                for idx, nsfw in enumerate(res.nsfw_content_detected):
                    if not nsfw:
                        generated_images.append(res.images[idx])
                return generated_images


class Image2ImageWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.server_generator == 'StableDiffusion':
            self.GenPipe = AutoPipelineForImage2Image.from_pretrained(
                'generator/stable-diffusion-v1-5', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                local_files_only=True, 
                use_safetensors=True,
            )
            if args.use_IPAdapter:
                self.GenPipe.load_ip_adapter(
                    "generator/IP-Adapter", 
                    subfolder="models", 
                    weight_name="ip-adapter_sd15.safetensors"
                )
        elif args.server_generator == 'StableDiffusionXL':
            self.GenPipe = AutoPipelineForImage2Image.from_pretrained(
                'generator/stable-diffusion-xl-base-1.0', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                local_files_only=True, 
                use_safetensors=True,
            )
            if args.use_IPAdapter:
                self.GenPipe.load_ip_adapter(
                    "generator/IP-Adapter", 
                    subfolder="sdxl_models", 
                    weight_name="ip-adapter_sdxl.safetensors"
                )
        elif args.server_generator == 'OpenJourney':
            self.GenPipe = AutoPipelineForImage2Image.from_pretrained(
                'generator/openjourney', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                local_files_only=True, 
                use_safetensors=True,
            )
        elif args.server_generator == 'FLUX':
            self.GenPipe = AutoPipelineForImage2Image.from_pretrained(
                'generator/FLUX.1-dev', 
                torch_dtype=torch.float16, 
                device_map='balanced', 
                max_memory={0:'10GB'}, 
                local_files_only=True, 
                use_safetensors=True,
            )
        else:
            raise NotImplementedError
        
        self.GenPipe.set_progress_bar_config(disable=True)
        if args.use_IPAdapter:
            self.GenPipe.set_ip_adapter_scale(args.IPAdapter_scale)
        self.img_size = max(512, args.img_size)
        self.transform = transforms.ToPILImage()


    def __call__(self, prompt, img, negative_prompt):
        with torch.no_grad():
            if img is None:
                image = self.transform(torch.rand(3, self.img_size, self.img_size)).convert("RGB")
            else:
                image = self.transform(img).resize((self.img_size, self.img_size)).convert("RGB")

            if self.args.server_generator == 'FLUX':
                res = self.GenPipe(prompt=prompt, 
                    image=image, 
                    strength=self.args.i2i_strength if img is not None else 1, 
                    height=self.img_size, 
                    width=self.img_size, 
                    num_images_per_prompt=self.args.num_images_per_prompt, 
                )
            else:
                res = self.GenPipe(prompt=prompt, 
                    image=image, 
                    strength=self.args.i2i_strength if img is not None else 1, 
                    negative_prompt=negative_prompt, 
                    height=self.img_size, 
                    width=self.img_size, 
                    num_images_per_prompt=self.args.num_images_per_prompt, 
                    ip_adapter_image=image if self.args.use_IPAdapter else None, 
                )

            if self.args.server_generator in ['StableDiffusionXL', 'FLUX']:
                return res.images
            else:
                generated_images = []
                for idx, nsfw in enumerate(res.nsfw_content_detected):
                    if not nsfw:
                        generated_images.append(res.images[idx])
                return generated_images
            

def get_generator(args):
    if args.task_mode == 'T2I':
        return Text2ImageWrapper(args)
    elif args.task_mode == 'I2I':
        return Image2ImageWrapper(args)
    else:
        raise NotImplementedError