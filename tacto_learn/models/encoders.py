import numpy as np
import torch as th
from torch import nn

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.
    Args:
      in_planes (int): The number of input feature maps.
      out_planes (int): The number of output feature maps.
      kernel_size (int): The filter size.
      dilation (int): The filter dilation factor.
      stride (int): The filter stride.
    """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )

class ImageEncoder(nn.Module):
    def __init__(self, input_shape, out_dim):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        assert len(input_shape) == 3 and input_shape[0] <= 4, f"Requires channel-first input shape but got {input_shape}"
        self.input_shape = input_shape      # channel-first
        self.out_dim = out_dim

        img_conv1 = conv2d(self.input_shape[0], 8, kernel_size=7, stride=2)
        img_conv2 = conv2d(8, 16, kernel_size=5, stride=2)
        img_conv3 = conv2d(16, 32, kernel_size=5, stride=2)
        # img_conv4 = conv2d(64, 64, kernel_size=3, stride=2)
        # img_conv5 = conv2d(64, 128, stride=2)
        # img_conv6 = conv2d(128, self.out_dim, stride=2)
        self.img_conv = nn.Sequential(
            img_conv1,
            img_conv2,
            img_conv3,
            # img_conv4,
            # img_conv5,
            # img_conv6,
        )

        self.flatten = nn.Flatten()
        cnn_out_dim = self.get_cnn_output_shape(self.input_shape)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, self.out_dim*2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.out_dim*2, self.out_dim)
        )

    def get_cnn_output_shape(self, image_shape):
        x = th.unsqueeze(th.rand(image_shape), dim=0)
        x = self.img_conv(x)
        return np.prod(x.shape)

    def forward(self, image):

        # image encoding layers
        x = self.img_conv(image)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class VectorEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.fc(x)