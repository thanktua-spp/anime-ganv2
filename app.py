from PIL import Image
import torch
import torchvision
import gradio

Instructuction = "Select a Unique Portrait Image of yourself"
title="I am something of a Painter, Anime-Edition (AnimeGAN-V2)"
description = "Drag/Drop or Upload a cute portrait Image of yourself or anyone you find interesting 😉, then observe how this Generative model\
               is able to Generate a cute Anime-Cartoon version of your Image."
article = """
            - Select an image from the examples provided as demo image,
            - Click submit to Generate Image,
            - Tips: Quality Images with Great brightness/without pre-existing filters work better (Image Noise).
            - Privacy: No user data is collected or saved,
            - Credits to akhaliq/AnimeGANv2 for original AnimeGanV2"""

model = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    #device="cuda",
    progress=False
)
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512, #device="cuda",
    side_by_side=False
)

def inference(img):
  return face2paint(model, img)

gradio.Interface(inference,
                 inputs=gradio.Image(type='pil'),
                 outputs=gradio.Image(type='pil'),
                 Instructuction=Instructuction, title=title, description=description, article=article,
                 examples=['Upload Xty.png',
                           'Upload Tg.png']).launch()
