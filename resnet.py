import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from activation import TReLU

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
model_parameters = {}
model_parameters['resnet18'] = [[64, 128, 256, 512], [2, 2, 2, 2], 1, False]
model_parameters['resnet34'] = [[64, 128, 256, 512], [3, 4, 6, 3], 1, False]
model_parameters['resnet50'] = [[64, 128, 256, 512], [3, 4, 6, 3], 4, True]
model_parameters['resnet101'] = [[64, 128, 256, 512], [3, 4, 23, 3], 4, True]
model_parameters['resnet152'] = [[64, 128, 256, 512], [3, 8, 36, 3], 4, True]

########################################################################################################################

class Bottleneck(nn.Module):

    def __init__(self, in_channels, intermediate_channels, expansion, is_Bottleneck, stride, depth, activation,
                 normalize, residual_connection):

        """
        Creates a Bottleneck with conv 1x1->3x3->1x1 layers.

        Note:
          1. Addition of feature maps occur at just before the final ReLU with the input feature maps
          2. if input size is different from output, select projected mapping or else identity mapping.
          3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Bottleneck is required for resnet-50/101/152
        Args:
            in_channels (int) : input channels to the Bottleneck
            intermediate_channels (int) : number of channels to 3x3 conv
            expansion (int) : factor by which the input #channels are increased
            stride (int) : stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining
            beta_init : initialization value for beta
            beta_is_trainable : true if beta is trainable, false otherwise

        """

        super(Bottleneck, self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        self.depth = depth
        self.activation = activation
        self.normalize = normalize
        self.residual_connection = residual_connection

        # i.e. if dim(x) == dim(F) => Identity function
        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels * self.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False))
            if self.normalize:
                projection_layer.append(nn.BatchNorm2d(self.intermediate_channels * self.expansion))
            # Only conv->BN
            self.projection = nn.Sequential(*projection_layer)

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

    def base_params(self):
        params = []

        if not self.identity:
            for param in self.projection.parameters():
                params.append(param)

        if self.is_Bottleneck:
            for param in self.conv1_1x1.parameters():
                params.append(param)
            if self.normalize:
                for param in self.batchnorm1.parameters():
                    params.append(param)
            for param in self.conv2_3x3.parameters():
                params.append(param)
            if self.normalize:
                for param in self.batchnorm2.parameters():
                    params.append(param)
            for param in self.conv3_1x1.parameters():
                params.append(param)
            if self.normalize:
                for param in self.batchnorm3.parameters():
                    params.append(param)
        else:
            for param in self.conv1_3x3.parameters():
                params.append(param)
            if self.normalize:
                for param in self.batchnorm1.parameters():
                    params.append(param)
            for param in self.conv2_3x3.parameters():
                params.append(param)
            if self.normalize:
                for param in self.batchnorm2.parameters():
                    params.append(param)

        return params

    def trelu_params(self):
        params = []
        if isinstance(self.activation, TReLU):
            for param in self.activation.parameters():
                params.append(param)
        return params

    def forward(self, x, beta=1.0):
        # input stored to be added before the final activation
        in_x = x

        if self.is_Bottleneck:
            # conv1x1->BN->activation
            if self.normalize:
                x = self.activation(self.batchnorm1(self.conv1_1x1(x)))
            else:
                x = self.activation(self.conv1_1x1(x))

            # conv3x3->BN->activation
            if self.normalize:
                x = self.activation(self.batchnorm2(self.conv2_3x3(x)))
            else:
                x = self.activation(self.conv2_3x3(x))

            # conv1x1->BN
            if self.normalize:
                x = self.batchnorm3(self.conv3_1x1(x))
            else:
                x = self.conv3_1x1(x)

        else:
            # conv3x3->BN->activation
            if self.normalize:
                x = self.activation(self.batchnorm1(self.conv1_3x3(x)))
            else:
                x = self.activation(self.conv1_3x3(x))

            # conv3x3->BN
            if self.normalize:
                x = self.batchnorm2(self.conv2_3x3(x))
            else:
                x = self.conv2_3x3(x)

        # scaling with beta for the residual connection
        if self.residual_connection:
            x = x * beta / np.sqrt(self.depth)
            # identity or projected mapping
            if self.identity:
                x += in_x
            else:
                x += self.projection(in_x)

        # final activation
        x = self.activation(x)

        return x

########################################################################################################################

class ResNet(nn.Module):

    def __init__(self,
                 resnet_variant,
                 in_channels,
                 num_classes,
                 activation,
                 alpha_init=1.0,
                 train_trelu=True,
                 residual_connections=True,
                 beta_init=0.5,
                 beta_is_trainable=True,
                 beta_is_global=False,
                 normalize=True):

        """
        Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repeatition list along with expansion factor(4) and stride(3/1)
        using _make_blocks method, create a sequence of multiple Bottlenecks
        Average Pool at the end before the FC layer

        Args:
            resnet_variant (str) : eg. 'resnet18', 'resnet34', 'resnet50' or 'resnet101'
            in_channels (int) : image channels (3)
            num_classes (int) : output #classes
            activation (str) : 'relu' or 'trelu'

        """
        super(ResNet, self).__init__()

        print(f"{'=' * 40}\n{'Model Settings':^40}\n{'=' * 40}")
        print(f"ResNet Variant: {resnet_variant}")
        print(f"In Channels: {in_channels}")
        print(f"Number of Classes: {num_classes}")
        print(f"Activation Function: {activation}")

        if activation == "trelu":
            print(f"TReLU Alpha Initialization: {alpha_init}")
            if train_trelu:
                print("TReLU is trainable.")
        elif activation == "relu":
            print("ReLU activation")

        if residual_connections:
            print("Residual connections on.")
            print(f"Residual Connections Beta Initialization: {beta_init}")
            if beta_is_trainable:
                print("Beta is trainable.")
                if beta_is_global:
                    print("Beta is global.")
                else:
                    print("Beta is per layer.")
        else:
            print("Residual connections off.")

        if normalize:
            print("Batch normalization on.")
        else:
            print("Batch normalization off.")

        print('=' * 40)

        resnet_parameters = model_parameters[resnet_variant]
        self.channels_list = resnet_parameters[0]
        self.repeatition_list = resnet_parameters[1]
        self.expansion = resnet_parameters[2]
        self.is_Bottleneck = resnet_parameters[3]
        self.device = device
        self.beta_is_global = beta_is_global
        self.residual_connections = residual_connections
        self.normalize = normalize
        self.depth = sum(self.repeatition_list)

        layers = []

        self.beta = None
        if residual_connections:
            if beta_is_global:
                self.beta = nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable)
            else:
                self.beta = nn.ParameterList(
                    [nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable) for _ in range(self.depth)])

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False))
        if self.normalize:
            layers.append(nn.BatchNorm2d(64))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "trelu":
            self.activation = TReLU(alpha_init, trainable=train_trelu, device=self.device)

        layers.append(self.activation)

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(self._make_blocks(in_channels=64,
                                        intermediate_channels=self.channels_list[0],
                                        num_repeat=self.repeatition_list[0],
                                        expansion=self.expansion,
                                        is_Bottleneck=self.is_Bottleneck,
                                        stride=1,
                                        depth=self.depth,
                                        activation=self.activation,
                                        normalize=normalize))

        layers.append(self._make_blocks(in_channels=self.channels_list[0] * self.expansion,
                                        intermediate_channels=self.channels_list[1],
                                        num_repeat=self.repeatition_list[1],
                                        expansion=self.expansion,
                                        is_Bottleneck=self.is_Bottleneck,
                                        stride=2,
                                        depth=self.depth,
                                        activation=self.activation,
                                        normalize=self.normalize))

        layers.append(self._make_blocks(in_channels=self.channels_list[1] * self.expansion,
                                        intermediate_channels=self.channels_list[2],
                                        num_repeat=self.repeatition_list[2],
                                        expansion=self.expansion,
                                        is_Bottleneck=self.is_Bottleneck,
                                        stride=2,
                                        depth=self.depth,
                                        activation=self.activation,
                                        normalize=self.normalize))

        layers.append(self._make_blocks(in_channels=self.channels_list[2] * self.expansion,
                                        intermediate_channels=self.channels_list[3],
                                        num_repeat=self.repeatition_list[3],
                                        expansion=self.expansion,
                                        is_Bottleneck=self.is_Bottleneck,
                                        stride=2,
                                        depth=self.depth,
                                        activation=self.activation,
                                        normalize=self.normalize))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten(start_dim=1))
        layers.append(nn.Linear(self.channels_list[3] * self.expansion, num_classes))

        self.net = nn.ModuleList(layers)

    def _make_blocks(self, in_channels, intermediate_channels, num_repeat, expansion,
                     is_Bottleneck, stride, depth, activation, normalize):
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
        self.depth = depth

        layers = []

        layers.append(
            Bottleneck(in_channels=in_channels, intermediate_channels=intermediate_channels, expansion=expansion,
                       is_Bottleneck=is_Bottleneck, stride=stride, depth=self.depth, activation=activation,
                       normalize=normalize, residual_connection=self.residual_connections))

        for num in range(1, num_repeat):
            layers.append(
                Bottleneck(in_channels=intermediate_channels * expansion, intermediate_channels=intermediate_channels,
                           expansion=expansion, is_Bottleneck=is_Bottleneck, stride=1, depth=self.depth,
                           activation=activation, normalize=normalize, residual_connection=self.residual_connections))

        return nn.Sequential(*layers)

    def base_params(self):
        params = []
        for l in self.net:
            if isinstance(l, Bottleneck):
                params += l.base_params()
            elif isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm1d):
                for param in l.parameters():
                    params.append(param)
        return params


    def trelu_params(self):
        params = []
        for l in self.net:
            if isinstance(l, Bottleneck):
                params += l.trelu_params()
            elif isinstance(l, TReLU):
                for param in l.parameters():
                    params.append(param)
        return params

    def forward(self, x):
        beta_idx = 0        # indexing the residual connection (one for each bottleck)

        for module in self.net:
            if self.residual_connections and isinstance(module, Bottleneck):
                beta = self.beta if self.beta_is_global else self.beta[beta_idx]
                beta_idx += 1
                x = module(x, beta=beta, depth_scaler=np.sqrt(self.depth))

            else:
                x = module(x)

        return x
