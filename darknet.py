import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union


class YOLOBlock(nn.Module):
    """
    Detection block for YOLOv3.
    """

    def __init__(self, anchors: List[Tuple[int, int]], n_classes: int = 80, training: bool = False) -> None:
        """
        Instantiates YOLOBlock (detection) object.

        :param anchors: Anchors for predefined bounding boxes.
        :param n_classes: Number of classes represented in data.
        :param arc: Dictates if the network used CE, BCE, or default.
        :param training: Whether to set to training mode.
        """
        super(YOLOBlock, self).__init__()

        self.anchors = torch.tensor(anchors)
        self.n_anchors = len(anchors)  # 3 per yolo layer
        self.n_classes = n_classes  # 80 if COCO
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

        self.training = training


    def forward(self, x: torch.Tensor, img_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw network output into correct box coordinates and class predictions for bounding
        boxes given network output.

        :param pred: Network output (raw prediction).
        :param img_size: Size of input image.
        :return: Inference and raw prediction.
        """
        # get batch size, height, and width from prediction
        batch_size = x.size()[0]
        ny, nx = x.size()[-2:]

        # if trying to perform detection at different scale than previously
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), x.device, x.dtype)

        # want in form batch_size x 3 (anchors per scale) x N (height) x N (width) x (80 (classes) + 4 (bounding box offsets) + 1 (objectness))
        #   - objectness is 1 if bounding overlaps ground truth object more than any other bounding box
        x = x.view(batch_size, self.n_anchors, self.n_classes + 4 + 1, self.ny, self.nx
                         ).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return x

        pred = x.clone()
        # calculate box offsets from network output
        # use sigmoid to scale coordinates down into a range in which they represent
        # percent of the grid cell into which they exist, from top left corner
        pred[..., 0:2] = torch.sigmoid(pred[..., 0:2]) + self.grid_xy
        pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.anchor_wh.float()
        pred[..., :4] *= self.stride

        torch.sigmoid_(pred[..., 4:])

        if self.n_classes == 1:
            pred[..., 5] = 1

        # reshape to batch size x ... x (80 (classes) + 4 (bounding box offsets) + 1 (objectness))
        return pred.view(batch_size, -1, 4 + 1 + self.n_classes), x

def create_grids(self, img_size: Tuple[int, int] = (416, 416), grid_dims: Tuple[int, int] = (13, 13), device: str = 'cpu',
                 type: type = torch.float32) -> None:
    """
    Breaks image into grid of equally sized cells. Will occur at three different scales.

    :param img_size: Size of input image.
    :param grid_dims: How many cells are making up the grid.
    :param device: GPU or CPU.
    :param type: Data type.
    """
    nx, ny = grid_dims
    self.img_size = max(img_size)
    self.stride = self.img_size / max(grid_dims)

    # xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view(1, 1, ny, nx, 2)

    # wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.n_anchors, 1, 1, 2).to(device)
    self.grid_dims = torch.tensor(grid_dims).to(device)
    self.nx = nx
    self.ny = ny


def conv_block(channels_in: int, n_filters: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    """
    Convolution->Batch Normalization->LeakyReLU. Core to Darknet feature extractor architecture.

    :param channels_in: Number of input channels to layer. i.e. number of output channels from previous layer.
    :param n_filters: Number of filters used by layer. Will be number of output channels.
    :param kernel_size: Size of filters, to be read as (kernel_size x kernel_size).
    :param stride: Step size of kernel across input.
    :return: nn.Sequential containing layers.
    """
    return nn.Sequential(
        nn.Conv2d(channels_in, n_filters, kernel_size=kernel_size, bias=False, stride=stride, padding=kernel_size // 2),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


class RouteBlock(nn.Module):
    """
    Concatenates output from one layer to another.
    """

    def __init__(self, routes: Tuple[int]) -> None:
        """
        Instantiates RouteBlock object: serves as dummy block for network routing.

        :param routes: Layers to connect.
        """
        super(RouteBlock, self).__init__()
        self.routes = routes

class Upsample(nn.Module):
    """
    Upsamples input using torch.nn.functional.interpolate
    """
    def forward(self, x):
        """
        Upsamples feature maps. nn.Upsample is deprecated.

        :param x: Input feature maps.
        :return: Upsampled feature maps.
        """
        return F.interpolate(x, scale_factor=2, mode='nearest')


class ResBlock(nn.Module):
    """
    Residual style block comprised of two conv blocks.
    """

    def __init__(self, channels_in: int, res_connection: bool = True) -> None:
        """
        Instantiates ResBlock object.

        :param channels_in: Number of channels to first of the two conv blocks.
        :param res_connection: Whether or not to include the skip connection.
        """
        super(ResBlock, self).__init__()
        self.res_connection = res_connection

        self.conv1 = conv_block(channels_in, channels_in // 2, kernel_size=1)
        self.conv2 = conv_block(channels_in // 2, channels_in, kernel_size=3)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], any]:
        """
        Forward pass of block.

        :param x: Input to block.
        :return: Processed output.
        """
        c1_out = self.conv1(x)
        c2_out = self.conv2(c1_out)

        # return as list so outputs can be easily added to cached output
        if self.res_connection:
            return [c1_out, c2_out], x + c2_out
        else:
            return [c1_out, c2_out]

class Darknet(nn.Module):
    """
    Feature extractor and detector from https://github.com/pjreddie/darknet.
    """

    def __init__(self, n_blocks: List[int], anchors: List[Tuple[int, int]],
                 routes: List[Tuple[int, int, int]], n_filters: int = 32,
                 n_detect: int = 3, fe_to_yolo: int = 3) -> None:
        """
        Instantiates Darknet feature extractor object.

        :param n_blocks: Each list element corresponds to the number of ResBlocks in one "chunk" of the network.
        :param anchors: Full list of anchors for predefined bounding boxes at each scale.
        :param n_filters: Number of filters off of which the first "chunk" of the network will be built.
        :param fe_to_yolo: Number of blocks in between the feature extractor and the yolo layers
        """
        super(Darknet, self).__init__()

        blocks = [conv_block(3, n_filters, kernel_size=3, stride=1)]

        for i, nb in enumerate(n_blocks):
            blocks += self.make_chunk(n_filters, nb, stride=2)
            n_filters *= 2

        blocks += self.make_detector(anchors, routes, n_filters, n_detect, fe_to_yolo)

        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for Darknet.

        :param x: Input to network.
        :return: Output from network.
        """
        img_size = x.size()[-2:]
        outputs = []        # cache outputs for route layers
        yolo_out = []

        for layer in self.layers:
            if isinstance(layer, RouteBlock):
                if len(layer.routes) == 1:
                    x = outputs[layer.routes[0]]
                else:
                    x = torch.cat(tuple([outputs[i] for i in layer.routes]), 1)
            elif isinstance(layer, YOLOBlock):
                x = layer(x, img_size)
                yolo_out.append(x[0])
            elif isinstance(layer, ResBlock):
                (c, x) = layer(x)
                if isinstance(c, list):
                    outputs += c
                else:
                    outputs.append(c)
            else:
                x = layer(x)
            outputs.append(x)

        return yolo_out

    def make_chunk(self, channels_in: int, n_blocks: int, stride: int = 2) -> List[Union[nn.Sequential, nn.Module]]:
        """
        Defines "chunk" of Darknet architecture (conv_block + nxResBlock).

        :param channels_in: Number of channels off of which filter numbers for "chunk" are defined.
        :param n_blocks: Number of ResBlocks that will sit in this "chunk."
        :param stride: Stride to be used by conv blocks.
        :return: Concatenated conv block and ResBlock modules.
        """
        return [conv_block(channels_in, channels_in * 2, stride=stride)
                ] + [ResBlock(channels_in * 2) for _ in range(n_blocks)]

    def make_detector(self, anchors: List[Tuple[int, int]], routes: List[Tuple[int, int, int]],
                      n_filters: int = 1024, n_detect: int = 3, fe_to_yolo: int = 3) -> List[nn.Module]:
        """
        Creates the part of the network responsible foe making predictions.

        :param anchors: Defaults for box locations.
        :param routes: Layer outputs to be concatenated.
        :param n_filters: Filters to be used to convolutional layers.
        :param n_detect: How many YOLO layers are to be used.
        :param fe_to_yolo: Layers in between the feature extractor and the YOLO layers.
        :return: Group of layers making up the detection part of the network.
        """
        blocks = []

        for i in range(n_detect):
            blocks += [ResBlock(channels_in=n_filters, res_connection=False) for _ in range(fe_to_yolo)
                        ] + [nn.Conv2d(n_filters, 255, kernel_size=1, bias=False, stride=1, padding=1)
                            ] + [YOLOBlock(anchors[len(anchors) - 3 * (i + 1):len(anchors) - 3 * i])]

            if i < n_detect - 1:
                # output will be cached for each network run to provide signal for route layers
                blocks += [RouteBlock(routes[i][:1])
                           ] + [conv_block(n_filters // 2, n_filters // 4, kernel_size=1, stride=1)
                                ] + [Upsample()
                                     ] + [RouteBlock(routes[i][1:])
                                          ] + [conv_block(n_filters - n_filters // 4, n_filters // 4, kernel_size=1, stride=1)
                                               ] + [conv_block(n_filters // 4, n_filters // 2, kernel_size=1, stride=1)]

            fe_to_yolo = 2
            n_filters = n_filters // 2

        return blocks
