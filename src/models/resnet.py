"""
Defines the ResNet-18 architecture adapted for CIFAR-10

The torchvision.models.resnet18 implementation is designed for
ImageNet, where images have a resolution of 224×224 pixels. For this
reason, the first convolutional layer uses a large 7×7 kernel with
stride 2, followed by a max-pooling layer. Together, these operations
significantly reduce the spatial resolution before the image reaches
the residual blocks

CIFAR-10 images are only 32×32 pixels. If the original ResNet is used
without modification, by the time the image reaches the second residual
block, very little spatial information remains. The feature maps become
too small too quickly, causing the network to lose valuable information

The standard adaptation consists of replacing the initial 7×7 convolution with a 3×3 convolution
using stride 1 and removing the initial max-pooling layer by replacing it with an
identity layer, which preserves the model structure without
altering the input

The activations from the penultimate layer will later be extracted to build
the feature manifold. Preserving as much information as possible from
the original 32×32 images leads to richer and more meaningful feature
representations
"""

import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Creates a ResNet-18 model adapted for CIFAR-10.

    Args:
        num_classes: Number of output classes (10 for CIFAR-10)
        pretrained: If True, loads ImageNet pretrained weights before
            adapting the first convolutional layer. For the supervised
            learning scenario, it is generally preferable to keep this
            set to False and train the network from scratch. This ensures
            that the feature manifold depends only on what the model
            learns from CIFAR-10, without any influence from ImageNet,
            making the comparison with the memorization and SimCLR
            scenarios cleaner and more meaningful

    Returns:
        A nn.Module model ready for training
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Adaptation for small (32×32) images
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()  # Remove the initial max-pooling layer

    # Classification head
    # "fc" is the final fully connected layer. We replace it so that
    # its output dimension matches the number of classes in our
    # classification task (10 for CIFAR-10). The number of input
    # features remains unchanged
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model