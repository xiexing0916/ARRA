import os
import sys
from transformers import AutoModel, AutoProcessor
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
from PIL import Image
from argparse import ArgumentParser
import json
import torch

def extract_paths_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_paths = [item["image"] for item in data]
    human_values = []
    for item in data:
        for convo in item.get("conversations", []):
            if convo.get("from") == "human":
                human_values.append([convo.get("value")])
    return image_paths, human_values

def extract_and_save_features_separately_from_json(json_path, save_dir="features"):
    os.makedirs(save_dir, exist_ok=True)
    image_texts = extract_paths_from_json(json_path)


    model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    preprocess = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    i = 0
    for [image_path, text] in zip(image_texts[0], image_texts[1]):
        image_input = Image.open(image_path[0]).convert("RGB")
        inputs = preprocess(images=image_input, return_tensors="pt", padding=True).to(device)
        # input_texts = preprocess(text=text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
            # text_embeddings = model.get_text_features(**input_texts)

        filename = os.path.basename(image_path[0]).split('.')[0] + "_visual_feature.pt"
        feature_path = os.path.join(save_dir, filename)

        torch.save(image_embeddings, feature_path)
        # print(img_feature.shape)  # [1,512]
        print(f"{i}: Feature for {image_path[0]} saved to {feature_path}")
        i = i + 1



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--in_filename",
        default="./MIMIC-CXR/xray_data.json",  # your dataset json
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="./pre_tokenization/cxr/biomedclip_feature",
        type=str,
    )
    args = parser.parse_args()

    json_path = args.in_filename
    extract_and_save_features_separately_from_json(json_path, save_dir=args.out_dir)




