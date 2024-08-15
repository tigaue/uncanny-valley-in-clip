import os
import clip
import torch
from PIL import Image
import glob

def calculate_image_score_clip(model, image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

if __name__ == '__main__':

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    dir_img = 'img'
    dir_out = 'output'
    list_method  = ['morph', 'superimposition', 'juxtaposition']

    for name_blend in list_method:

        print('loading images from ' + dir_img + '/' + name_blend + '/*.png' + '...')
        list_path = {}
        list_path = sorted(glob.glob(dir_img + '/' + name_blend + '/*.png'))
        dict_img = {}
        for i, path in enumerate(list_path):
            if i % 100 == 0:
                print('\r' + str(i) + '/' + str(len(list_path)), end="")
            img = Image.open(path).copy()
            name = path.split('/')[-1]
            dict_img[name] = img
        print()
        
        print('calculating embeddings...')
        list_emb_norm = {}
        for key in dict_img:
            img = dict_img[key]
            img_score = calculate_image_score_clip(model, img)
            list_emb_norm[key] = img_score
            
        print('saving...')
        dir_save = dir_out + '/' + name_blend
        os.makedirs(dir_save)
        path_save = dir_save + '/' + 'clip_embedding_normalized.pt'
        torch.save(list_emb_norm, path_save)

