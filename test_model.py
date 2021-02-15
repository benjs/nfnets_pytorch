import torch
from model import SqueezeExcite, NFBlock

class TestSqueezeExcite:
    def test_shape(self):
        ch_in = 10
        ch_out = ch_in

        x = torch.randn(5, ch_in, 100, 100) #NCHW
        se = SqueezeExcite(ch_in, ch_out)
        y = se(x)

        assert x.size() == y.size()

class TestNFBlock:
    def test_shape(self):
        channels = [256, 512, 1536]

        for ch in channels:
            block = NFBlock(in_channels=ch//2, out_channels=ch, stride=1)

            h,w = 200, 200
            b = 2
            x = torch.randn(b, ch // 2, h, w)
            
            y = block.conv0(x)
            assert y.size()[0] == b
            assert y.size()[1] == ch // 2
            assert y.size()[2:4] == torch.Size([h, w])

            y = block.conv1(y)
            assert y.size()[0] == b
            assert y.size()[1] == ch / 2
            assert y.size()[2:4] == torch.Size([h, w])

            y = block.conv1b(y)
            assert y.size()[0] == b
            assert y.size()[1] == ch // 2
            assert y.size()[2:4] == torch.Size([h, w])

            y = block.conv2(y)
            assert y.size()[0] == b
            assert y.size()[1] == ch
            assert y.size()[2:4] == torch.Size([h, w])

            y = block.se(y)
            assert y.size()[0] == b
            assert y.size()[1] == ch
            assert y.size()[2:4] == torch.Size([h, w])


            