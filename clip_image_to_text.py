from PIL import Image
from clip_interrogator import Config, Interrogator

def load_interrogator(model_path, caption_model_name="blip-base", device="cpu"):
    clip_intrerrogator = Interrogator(
        Config(caption_model_name=caption_model_name, 
               clip_model_name="ViT-L-14/openai", 
               cache_path=model_path, 
               clip_model_path=model_path,
               quiet=True,
               device=device)
        )

    return clip_intrerrogator.interrogate


def inference(img_pil, model):
    result = model(img_pil)
    return result
    

if __name__ == "__main__":
    img_pil = Image.open("/media/mlfavorfit/sdb/contolnet_dataset/val/158_Chane.jpg").convert('RGB')
    
    model = load_interrogator("./temp", caption_model_name="blip-base", device="cuda") 
    result = inference(img_pil=img_pil, model=model)
    print(result)
