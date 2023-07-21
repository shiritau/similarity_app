import logging
import torch
import numpy as np
from numpy.linalg import norm
import streamlit as st
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mtcnn = MTCNN(image_size=224, post_process=False)


def get_vgg_model(weights_path=None):
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model

def get_similarities(img1, img2, perform_mtcnn=True, weights = None):
    features1 = get_embedding_single_image(img1, perform_mtcnn, weights)
    features2 = get_embedding_single_image(img2, perform_mtcnn, weights)

    first_em = torch.flatten(features1).detach().numpy()
    second_em = torch.flatten(features2).detach().numpy()
    return np.dot(first_em, second_em) / (norm(first_em) * norm(second_em))


def get_embedding_single_image(img, perform_mtcnn=True, weights = None):
    model = get_vgg_model(weights)
    if img.mode != 'RGB':  # PNG imgs are RGBA
        img = img.convert('RGB')
    if perform_mtcnn:
        img = mtcnn(img)
        # if mtcnn post_processing = true, image is standardised to -1 to 1, and size 160x160.
        # else, image is not standardised ([0-255]), and size is 160x160
        try:
            img = img / 255.0
        except TypeError:
            logging.info(f'img is {type(img)}')

        img = normalize_imagenet(img)
        features = model(img.unsqueeze(0).float())
        return features