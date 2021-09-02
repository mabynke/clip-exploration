import os
import shutil

import numpy as np
import torch

import clip
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

filetternavn = ("png", "jpg", "jpeg")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(f"device: {device}")

dir = input("Mappe å søke i: ")
tekst = input("Streng å sammenligne bildet med: ")
text = clip.tokenize([tekst]).to(device)
print("resultat av tokenize:", text, sep="\n")

for bane, mapper, filer in os.walk(dir):
    print("Går inn i mappe:", bane)
    
    bildefiler = filter(lambda filnavn : filnavn.split(".")[-1] in filetternavn, filer)
    bildefiler = list(bildefiler)
    if not bildefiler:
        continue
    mellomlagringsbane = os.path.join("/tmp/bilder", "mellomlager", bane.replace(os.path.sep, "_") + ".pt")
    mellomlagret = os.path.isfile(mellomlagringsbane) 
    if not mellomlagret:
        print("Ikke mellomlagret fra før, ordner ...")
        bilder = [preprocess(Image.open(os.path.join(bane, bildefil))) for bildefil in bildefiler]
        bilder = torch.stack(bilder).to(device)
        
        image_features = None
        print(bilder.shape)

    print("Kjører modellen ...")
    with torch.no_grad():
        if mellomlagret:
            print("Laster bildevektorer fra mellomlager:", mellomlagringsbane)
            image_features = torch.load(mellomlagringsbane)
        else:
            image_features = model.encode_image(bilder)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            print("Mellomlagrer ...")
            torch.save(image_features, mellomlagringsbane)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print("image_features.shape:", image_features.shape)
        print("text_features.shape:", text_features.shape)
        likheter = 100 * image_features @ text_features.t()
        
        
        #logits_per_image, logits_per_text = model(bilder, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        #print("logits_per_image:", logits_per_image, sep="\n")
        #print("likheter:", likheter, sep="\n")

    for i in range(likheter.shape[0]):
        if not (15 <= likheter[i][0] <= 24):
            print(likheter[i].cpu().numpy(), bildefiler[i])
            os.symlink(os.path.join(bane, bildefiler[i]), os.path.join("/tmp/bilder", str(100 * likheter[i][0].cpu().item()) + "_" + bildefiler[i]))
            #print("UTVALGT")
        # p = subprocess.Popen(["eog", os.path.join(bane, bildefiler[i])])
        # p.wait()
