import math

import torch
import torch.nn as nn

from typing import Optional

########################################################################################################################

# set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

########################################################################################################################

# resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
model_parameters={}
model_parameters['resnet18'] = [[64, 128, 256, 512], [2, 2, 2, 2], 1, False]
model_parameters['resnet34'] = [[64, 128, 256, 512], [3, 4, 6, 3], 1, False]
model_parameters['resnet50'] = [[64, 128, 256, 512], [3, 4, 6, 3], 4, True]
model_parameters['resnet101'] = [[64, 128, 256, 512], [3, 4, 23, 3], 4, True]
model_parameters['resnet152'] = [[64, 128, 256, 512], [3, 8, 36, 3], 4, True]

########################################################################################################################

class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 expansion,
                 is_Bottleneck,
                 stride,
                 use_batch_norm,
                 shortcut_weight,
                 activation
                 ):

        """
        Creates a Bottleneck with conv 1x1->3x3->1x1 layers.

        Note:
          1. Addition of feature maps occur at just before the final ReLU with the input feature maps
          2. if input size is different from output, select projected mapping or else identity mapping.
          3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Required for resnet-50/101/152
        Args:
            in_channels (int) : input channels to the Bottleneck
            intermediate_channels (int) : number of channels to 3x3 conv
            expansion (int) : factor by which the input #channels are increased
            stride (int) : stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining
            use_batch_norm (bool) : whether to use Batch Normalization (BN).
            shortcut_weight: the weighting factor of shortcut branch, which must be a float between 0 and 1, or None.
                If not None, the shortcut branch is multiplied by "shortcut_weight", and the residual branch is
                multiplied by "residual_weight", where
                        "shortcut_weight**2 + residual_weight**2 == 1.0".
                If None, no multiplications are performed (which corresponds to a standard ResNet). Note that setting
                "shortcut_weight" to 0.0 effectively removes the skip connections from the network.
            activation: activation defined in according to activation name in ResNet
        """

        super(Bottleneck, self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        self.use_batch_norm = use_batch_norm
        self.shortcut_weight = shortcut_weight
        self.activation = activation

        # i.e. if dim(x) == dim(F) => Identity function
        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            # Only conv->BN and no activation
            self.identity = False
            projection_layer = []
            projection_layer.append(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels * self.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False))
            if use_batch_norm:
                projection_layer.append(nn.BatchNorm2d(self.intermediate_channels * self.expansion))
                self.projection = nn.Sequential(*projection_layer)
            else:
                self.projection = nn.Sequential(*projection_layer)

        self.activation = activation

        # is_Bottleneck = True for all ResNet 50+
        if self.is_Bottleneck:
            # bottleneck
            # 1x1
            self.conv1_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels,
                                       kernel_size=1, stride=1, padding=0, bias=False)
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)

            # 3x3
            self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels,
                                       kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)

            # 1x1
            self.conv3_1x1 = nn.Conv2d(in_channels=self.intermediate_channels,
                                       out_channels=self.intermediate_channels * self.expansion, kernel_size=1,
                                       stride=1, padding=0, bias=False)
            self.batchnorm3 = nn.BatchNorm2d(self.intermediate_channels * self.expansion)

        else:
            # basicblock
            # 3x3
            self.conv1_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels,
                                       kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchnorm1 = nn.BatchNorm2d(self.intermediate_channels)

            # 3x3
            self.conv2_3x3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels,
                                       kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm2 = nn.BatchNorm2d(self.intermediate_channels)

    def forward(self, x):
        # input stored to be added before the final activation
        in_x = x

        if self.is_Bottleneck:

            # conv1x1->BN->activation
            x = self.conv1_1x1(x)
            if self.use_batch_norm:
                x = self.batchnorm1(x)
            x = self.activation(x)

            # conv3x3->BN->activation
            x = self.conv2_3x3(x)
            if self.use_batch_norm:
                x = self.batchnorm2(x)
            x = self.activation(x)

            # conv1x1->activation
            x = self.conv3_1x1(x)
            if self.use_batch_norm:
                x = self.batchnorm3(x)

        else:
            # conv3x3->BN->activation
            x = self.conv1_3x3(x)
            if self.use_batch_norm:
                x = self.batchnorm1(x)
            x = self.activation(x)

            # conv3x3->BN
            x = self.conv2_3x3(x)
            if self.use_batch_norm:
                x = self.batchnorm2(x)


        if self.identity:
            if self.shortcut_weight is None:
                x += in_x
            elif self.shortcut_weight != 0.0:
                x = math.sqrt(1 - self.shortcut_weight ** 2) * x + self.shortcut_weight * in_x
            else:
                # unchanged x
                pass

        else:
            if self.shortcut_weight is None:
                x += self.projection(in_x)
            elif self.shortcut_weight != 0.0:
                x = math.sqrt(1 - self.shortcut_weight ** 2) * x + self.projection(self.shortcut_weight * in_x)
            else:
                # unchanged x
                pass

        # final activation
        x = self.activation(x)

        return x

########################################################################################################################

class ResNet(nn.Module):

    def __init__(self,
                 resnet_variant: list,
                 in_channels: int,
                 num_classes: int,
                 use_batch_norm: bool = True,
                 shortcut_weight: Optional[float] = None,
                 activation_name: str = "relu",
                 ):
        """
        Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repeatition list along with expansion factor(4) and
        stride(3/1) using _make_blocks method, create a sequence of multiple Bottlenecks Average Pool at the end before
        the FC layer

        Args:
            resnet_variant (list) : eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_channels (int) : image channels (3)
            num_classes (int) : output #classes
            use_batch_norm (bool) : whether to use Batch Normalization (BN). Defaults to "True".
            shortcut_weight: the weighting factor of shortcut branch, which must be a float between 0 and 1, or None.
                If not None, the shortcut branch is multiplied by "shortcut_weight", and the residual branch is
                multiplied by "residual_weight", where
                        "shortcut_weight**2 + residual_weight**2 == 1.0".
                If None, no multiplications are performed (which corresponds to a standard ResNet). Note that setting
                "shortcut_weight" to 0.0 effectively removes the skip connections from the network. Defaults to None.
            activation_name: string name for activation function. Can be "leaky_relu" or "relu". Defaults to "relu".
        """
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]
        self.use_batch_norm = use_batch_norm
        self.shortcut_weight = shortcut_weight
        self.activation_name = activation_name

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        self.batchnorm1 = nn.BatchNorm2d(64)

        if self.activation_name == "relu":
            self.activation = nn.ReLU()
        elif self.activation_name == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError("Invalid activation name!")

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_blocks(64, self.channels_list[0], self.repeatition_list[0], self.expansion,
                                        self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0] * self.expansion, self.channels_list[1],
                                        self.repeatition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1] * self.expansion, self.channels_list[2],
                                        self.repeatition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2] * self.expansion, self.channels_list[3],
                                        self.repeatition_list[3], self.expansion, self.is_Bottleneck, stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        if self.use_batch_norm:
            x = self.batchnorm1(x)

        x = self.activation(x)

        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.average_pool(x)

        if self.training and not self.use_batch_norm:
            x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

    def _make_blocks(self, in_channels, intermediate_channels, num_repeat, expansion, is_Bottleneck, stride):
        """
        Args:
            in_channels : #channels of the Bottleneck input
            intermediate_channels : #channels of the 3x3 in the Bottleneck
            num_repeat : #Bottlenecks in the block
            expansion : factor by which intermediate_channels are multiplied to create the output channels
            is_Bottleneck : status if Bottleneck in required
            stride : stride to be used in the first Bottleneck conv 3x3

        Attributes:
            Sequence of Bottleneck layers

        """

        layers = []

        layers.append(Bottleneck(in_channels, intermediate_channels, expansion, is_Bottleneck, stride=stride,
                                 use_batch_norm=self.use_batch_norm,
                                 shortcut_weight=self.shortcut_weight,
                                 activation=self.activation))
        for num in range(1, num_repeat):
            layers.append(Bottleneck(intermediate_channels * expansion, intermediate_channels, expansion, is_Bottleneck,
                                     stride=1, use_batch_norm=self.use_batch_norm,
                                     shortcut_weight=self.shortcut_weight,
                                     activation=self.activation))

        return nn.Sequential(*layers)
