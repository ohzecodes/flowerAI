from torchvision import models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def findModel(arch):

    dispatch = {
        "vgg11": lambda: models.vgg11(weights=None),
        "squeezenet": lambda: models.alexnet(weights=None),
        "resnet18": lambda: models.resnet18(pretrained=True),
        "alexnet": lambda: models.alexnet(pretrained=True),
        "vgg16": lambda: models.vgg16(),
        "densenet161": lambda: models.densenet161(pretrained=True),
        "inception": lambda: models.inception_v3(pretrained=True),
        "googlenet": lambda: models.googlenet(pretrained=True),
        "shufflenet": lambda: models.shufflenet_v2_x1_0(pretrained=True),
        "mobilenet": lambda: models.mobilenet_v2(pretrained=True),
        "resnext": lambda: models.resnext50_32x4d(pretrained=True),
        "wide_resnet": lambda: models.wide_resnet50_2(pretrained=True),
        "mnasnet": lambda: models.mnasnet1_0(pretrained=True)
    }
    if arch in dispatch:
        return dispatch[arch]()
    else:
        raise ValueError("Architecture not supported")
