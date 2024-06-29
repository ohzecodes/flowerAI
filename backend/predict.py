import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets as d
from torchvision import transforms as t
from torchvision.transforms import Compose as c

from PIL import Image

from FindModel import findModel
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", help="path to image")
    parser.add_argument("checkpoint", help="path to checkpoint")
    parser.add_argument("--top_k", default=5, help="the number of option you want to show")
    parser.add_argument("--category_name", default="cat_to_name.json", help="category names")
    parser.add_argument("--gpu", action='store_true', help="should I use gpu for predicting?")
    return parser.parse_args()


def get_device(gpu):
    return torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")


def load_model(checkpoint, device):
    stateDict = torch.load(checkpoint)
    model = findModel(stateDict["arch"])
    model.classifier = stateDict['classifier']
    model.load_state_dict(stateDict['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def load_flower_names(category_name):
    with open(category_name, 'r') as f:
        return json.load(f)


def get_test_dataset():
    test_data = c([
        t.Resize(255),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return d.ImageFolder('flowers/test', transform=test_data)


def process_image(image):
    filename = image
    img = Image.open(filename)
    transform = c([t.Resize(256),
                   t.CenterCrop(224),
                   t.ToTensor(),
                   t.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])])
    if img.mode != 'RGB':
        img = img.convert('RGB')
    timg = transform(img)
    timg = t.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])(timg)
    return np.array(timg)


def predict(model, image_path, top_k, device):
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img.to(device)
    output = model(img)
    probability = torch.exp(output)
    top_probability, top_indices = probability.topk(top_k)

    top_probability = top_probability.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    index_to_class = {value: key for key, value in get_test_dataset().class_to_idx.items()}

    top_flower_names = [index_to_class[index] for index in top_indices]

    return top_probability, top_flower_names


def output_stats(image_path, names_flowers, model, top_k, device):
    probability, flowerNames = predict(model, image_path, top_k, device)
    names = []
    for i in flowerNames:
        names += [names_flowers[i]]
    return list(zip(names, probability))


def display_stats(image_path, data):
    names = [item[0] for item in data]
    probability = [item[1] for item in data]
    image = Image.open(image_path)
    f, ax = plt.subplots(2, figsize=(6, 10))
    ax[0].imshow(image)
    ax[0].set_title(names[0])
    y_names = np.arange(len(names))
    ax[1].barh(y_names, probability)
    ax[1].set_yticks(y_names)
    ax[1].set_yticklabels(names)
    ax[1].invert_yaxis()
    plt.show()


def main_flask(image_path, checkpoint, top_k=5, category_name="cat_to_name.json", gpu=False):
    device = get_device(gpu)
    model = load_model(checkpoint, device)
    names_flowers_map = load_flower_names(category_name)
    prediction = output_stats(image_path, names_flowers_map, model, top_k, device)
    return prediction


def main():
    args = parse_args()

    device = get_device(args.gpu)
    model = load_model(args.checkpoint, device)
    print(model)
    names_flowers_map = load_flower_names(args.category_name)
    prediction = output_stats(args.path, names_flowers_map, model, args.top_k, device)

    display_stats(args.path, prediction)


if __name__ == "__main__":
    main()
