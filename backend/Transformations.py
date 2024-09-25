import torchvision.transforms as t


# Define the function to get transformations based on the model type
def get_transformations(model_name):
    if model_name == 'squeezenet':
        train_transform = t.Compose([
            t.RandomRotation(60),
            t.RandomResizedCrop(224),
            t.RandomHorizontalFlip(),
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # SqueezeNet normalization
        ])

        valid_transform = t.Compose([
            t.Resize(255),
            t.CenterCrop(224),  # Maintain the 224x224 size
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = t.Compose([
            t.Resize(255),
            t.CenterCrop(224),  # Maintain the 224x224 size
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif 'vgg' in model_name.lower():

        train_transform = t.Compose([
            t.RandomRotation(60),
            t.RandomResizedCrop(224),
            t.RandomHorizontalFlip(),
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        valid_transform = t.Compose([
            t.Resize(255),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = t.Compose([
            t.Resize(255),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    else:
        raise ValueError("Model name not recognized. Use 'squeezenet' or 'vgg'.")

    return train_transform, valid_transform, test_transform



