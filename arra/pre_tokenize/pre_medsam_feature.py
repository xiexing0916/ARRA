import os
import sys
from transformers import SamModel, SamProcessor
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
from PIL import Image
from argparse import ArgumentParser
import json
import torch
import torch.nn as nn

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



    model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
    processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gap = nn.AdaptiveAvgPool2d((1, 1))


    i = 0
    for [image_path, text] in zip(image_texts[0], image_texts[1]):
        image_input = Image.open(image_path[0]).convert("RGB")
        inputs = processor(images=image_input, return_tensors="pt", padding=True).to(device)

        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            vision_features = model.vision_encoder(pixel_values=pixel_values)
            vision_features = vision_features.last_hidden_state
            pooled_features = gap(vision_features).squeeze(-1).squeeze(-1)
            # image_embeddings = model.get_image_features(**inputs)

            # text_feature = model.text(texts)

        filename = os.path.basename(image_path[0]).split('.')[0] + "_visual_feature.pt"
        feature_path = os.path.join(save_dir, filename)

        torch.save(pooled_features, feature_path)
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
        default="./pre_tokenization/cxr/medsam_feature",
        type=str,
    )
    args = parser.parse_args()

    json_path = args.in_filename
    extract_and_save_features_separately_from_json(json_path, save_dir=args.out_dir)




