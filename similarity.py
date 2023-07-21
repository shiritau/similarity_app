import logging
import torch
import torchvision
from torch import nn
from torchvision import transforms
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from numpy.linalg import norm
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#composed_transforms = transforms.Compose([transforms.ToTensor(), Rescale((224, 224)), normalize_imagenet])


def get_vgg_model(weights_path = None):
    model =InceptionResnetV1().eval()
    #model.classifier[6] = nn.Linear(4096, 2000)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=False)


    return model

def get_vgg_pretrained_vggface2(weights_path, return_layer='classifier.4'):
    model = torchvision.models.vgg16().eval()
    model.features = torch.nn.DataParallel(model.features)
    model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=8749)
    weights = torch.load(weights_path, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(weights)
    return_layers = {return_layer: 'output'}
    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
    return mid_getter

weights = r"C:\Users\shiri\Documents\School\Master\Galit\epoch_60_weights.pt"
#weights = "/home/Shirialmog/mysite/models/epoch_60_weights.pt"
model = get_vgg_model(weights)
mtcnn = MTCNN(image_size=224, post_process=False)

def get_similarities(img1, img2):

    features1 = get_embedding_single_image(img1, perform_mtcnn=True)
    features2 = get_embedding_single_image(img2, perform_mtcnn=True)


    first_em = torch.flatten(features1).detach().numpy()
    second_em = torch.flatten(features2).detach().numpy()
    return  np.dot(first_em, second_em) / (norm(first_em) * norm(second_em))


def get_embedding_single_image(img, perform_mtcnn=True):
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
    # else:  # in case images are already post mtcnn. In this case need to rescale to 160x160 and normalize
    #     img = composed_transforms(img)
