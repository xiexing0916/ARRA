
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

def BioMedClip(images):
    model_name = "biomedclip_local"

    # with open(
    #         "/data/xxing/Lumina-mGPT/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json",
    #         "r") as f:
    #     config = json.load(f)
    #     model_cfg = config["model_cfg"]
    #     preprocess_cfg = config["preprocess_cfg"]
    #
    # if (not model_name.startswith(HF_HUB_PREFIX)
    #         and model_name not in _MODEL_CONFIGS
    #         and config is not None):
    #     _MODEL_CONFIGS[model_name] = model_cfg

    # tokenizer = get_tokenizer(model_name)

    # model, _, preprocess = create_model_and_transforms(
    #     model_name=model_name,
    #     pretrained="/data/xxing/Lumina-mGPT/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin",
    #     **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    # )

    all_texts = ["Pneumothorax", "Pleural Effusion", "Cardiomegaly", "Pneumonia", "Fracture", "No Finding"]
    target_texts = ["Pneumothorax", "Pleural Effusion", "Cardiomegaly", "Pneumonia", "Fracture", "No Finding"]

    model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    preprocess = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()


    inputs = preprocess(text=target_texts, return_tensors="pt", padding=True).to(device)
    images = [Image.open(image_path) for image_path in images]
    image_input = preprocess(images=images, return_tensors="pt", padding=True).to(device)
    image_input1 = image_input.data["pixel_values"]

    with torch.no_grad():
        image_output = model.vision_model(image_input1)
        last_hidden_state = image_output.last_hidden_state
        last_hidden_state = model.visual_projection(last_hidden_state)
        text_embeddings = model.get_text_features(**inputs)


    return last_hidden_state, text_embeddings

if __name__ == "__main__":
    test_imgs = [
        "./datasets/MIMIC-CXR/files/p10/p10003502/s52309364/e0275ad1-1e6a7451-c3960f5f-1267a188-547b73a1.jpg"
    ]
    feature = BioMedClip(test_imgs)
    print(feature[1].shape)
