from MobileNetV2_dynamicFPN import InvertedResidual
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1, padding=0, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ConvTransposeBN(nn.Module):
    def __init__(self, inp, oup, kernel_size=2, stride=2):
        super(ConvTransposeBN, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(inp, oup, kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv_transpose(x)

class Centerface(nn.Module):
    def __init__(self, width_mult=1.):
        super(Centerface, self).__init__()

        self.input_channel = int(32 * width_mult)
        self.first_output_channel = int(16 * width_mult)
        self.width_mult = width_mult

        # First layer_block
        self.first_block = nn.Sequential(
            ConvBN(3, self.input_channel, kernel_size=3, stride=2, padding=1),
            ConvBN(self.input_channel, self.input_channel, kernel_size=3, stride=1, padding=1,
             groups=self.input_channel),
            nn.Conv2d(self.input_channel, self.first_output_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.first_output_channel)
        )
        self.input_channel = self.first_output_channel

        # Inverted residual blocks (each n layers)
        self.inverted_residual_setting = [
            # {'expansion_factor': 1, 'width_factor': 16, 'n': 1, 'stride': 1}, #32 -> 16
            {'expansion_factor': 6, 'width_factor': 24, 'n': 2, 'stride': 2, 'lateral': True},
            {'expansion_factor': 6, 'width_factor': 32, 'n': 3, 'stride': 2, 'lateral': True},
            {'expansion_factor': 6, 'width_factor': 64, 'n': 4, 'stride': 2, 'lateral': False},
            {'expansion_factor': 6, 'width_factor': 96, 'n': 3, 'stride': 1, 'lateral': True},
            {'expansion_factor': 6, 'width_factor': 160, 'n': 3, 'stride': 2, 'lateral': False},
            {'expansion_factor': 6, 'width_factor': 320, 'n': 1, 'stride': 1, 'lateral': True},
        ]
        self.inverted_residual_blocks = nn.ModuleList(
            [self._make_inverted_residual_block(**setting)
             for setting in self.inverted_residual_setting])

        # reduce feature maps to one pixel
        # allows to upsample semantic information of every part of the image

        self.p_level_c = 24 # centerface pyramid levels channels

        # Top layer
        # input channels = last width factor
        self.top_layer = ConvBN(
            int(self.inverted_residual_setting[-1]['width_factor'] * self.width_mult),
            self.p_level_c, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        # exclude last setting as this lateral connection is the the top layer

        self.lateral_setting = [setting for setting in self.inverted_residual_setting[:-1]
                                if setting['lateral']]
        self.lateral_layers = nn.ModuleList([
            ConvBN(int(setting['width_factor'] * self.width_mult),
                    self.p_level_c, kernel_size=1, stride=1, padding=0)
            for setting in self.lateral_setting])
        
        # upsample using conv-transpose
        self.conv_transposes = nn.ModuleList([
            ConvTransposeBN(self.p_level_c, self.p_level_c, kernel_size=2, stride=2)
            for _ in range(len(self.lateral_layers))
        ])

        self.f_map = ConvBN(self.p_level_c, self.p_level_c, kernel_size=3, stride=1, padding=1)

        self.reg = nn.Conv2d(self.p_level_c, 1, kernel_size=1, stride=1)

        self.landmarks_reg = nn.Conv2d(self.p_level_c, 10, kernel_size=1, stride=1)
        self.heatmap_reg = nn.Sequential(
            nn.Conv2d(self.p_level_c, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.scales_reg = nn.Conv2d(self.p_level_c, 2, kernel_size=1, stride=1)
        self.offsets_reg = nn.Conv2d(self.p_level_c, 2, kernel_size=1, stride=1)


        # self._initialize_weights()

    def _make_inverted_residual_block(self, expansion_factor, width_factor, n, stride, lateral):
        inverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            # except the first layer, all layers have stride 1
            if i != 0:
                stride = 1
            inverted_residual_block.append(
                InvertedResidual(self.input_channel, output_channel, stride, expansion_factor))
            self.input_channel = output_channel

        return nn.Sequential(*inverted_residual_block)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # bottom up
        x = self.first_block(x)

        # loop through inverted_residual_blocks (mobile_netV2)
        # save lateral_connections to lateral_tensors
        # track how many lateral connections have been made

        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(x)  # run block of mobile_net_V2
            if self.inverted_residual_setting[i]['lateral'] \
                    and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append(self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            x = output

        y = self.top_layer(x)

        # reverse lateral tensor order for top down
        lateral_tensors.reverse()
        for i, lateral_tensor in enumerate(lateral_tensors):
            y = self.conv_transposes[i](y)
            y = y + lateral_tensor
                
        features_map = self.f_map(y)

        heatmap = self.heatmap_reg(features_map)
        scales = self.scales_reg(features_map)
        offsets = self.offsets_reg(features_map)
        landmarks = self.landmarks_reg(features_map)
        return heatmap, scales, offsets, landmarks

class Test(nn.Module):
    def __init__(self):
        super().__init__()

        self.p = nn.Conv2d(192, 192, 3, 1, padding=1)

    def forward(self, x):
        return self.p(x)

# t = Test()
# x = torch.randn(1,192,80,80)
# y = t(x)
# print(y.shape)
