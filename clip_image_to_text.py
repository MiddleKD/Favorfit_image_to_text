from PIL import Image
from clip_interrogator import Config, Interrogator
from torch import nn

class InterrogatorWrapper(nn.Module):
    def __init__(self, model_path, caption_model_name="blip-base", device="cpu"):
        super().__init__()
        self.clip_interrogator = Interrogator(
        Config(caption_model_name=caption_model_name, 
               clip_model_name="ViT-L-14/openai", 
               cache_path=model_path, 
               clip_model_path=model_path,
               quiet=True,
               device=device)
        )
    
    def forward(self, img_pil, mode="fast"):
        if mode=="fast":
            result = self.clip_interrogator.interrogate_fast(img_pil)
        elif mode=="classic":
            result = self.clip_interrogator.interrogate_classic(img_pil)
        elif mode=="negative":
            result = self.clip_interrogator.interrogate_negative(img_pil)
        else:
            result = self.clip_interrogator.interrogate(img_pil)
        return result


def load_interrogator(model_path, caption_model_name="blip-base", device="cpu"):
    clip_intrerrogator = InterrogatorWrapper(
        model_path=model_path,
        caption_model_name=caption_model_name, 
        device=device
    )
    return clip_intrerrogator


def inference(img_pil, model, mode="fast"):
    result = model(img_pil, mode=mode)
    return result
    

if __name__ == "__main__":
    img_pil = Image.open("/media/mlfavorfit/sdb/contolnet_dataset/val/158_Chane.jpg").convert('RGB')
    
    model = load_interrogator("/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/image_to_text/clip", caption_model_name="blip-base", device="cuda") 
    result = inference(img_pil=img_pil, model=model)
    print(result)
