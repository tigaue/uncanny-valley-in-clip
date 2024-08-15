import clip
import torch

import argparse
def parse_input():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_feat', type=str, default='output/morph/clip_embedding_normalized.pt')
    parser.add_argument('--name_img', type=str, default='fg-002-004_fg-005-004_mor-015_rot-000.png')
    parser.add_argument('--c', type=str, default='zombie')
    return parser.parse_args()

def calculate_text_feature_clip(model, c):
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}")]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

if __name__ == '__main__':
    args = parse_input()

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Load image features
    img_data = torch.load(args.path_feat)
    image_features = img_data[args.name_img]

    # Calculate text features
    text_features = calculate_text_feature_clip(model, args.c)

    # Output cosine similarity
    similarity = image_features @ text_features.T
    print(similarity)
