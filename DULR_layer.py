import torch
import torch.nn as nn
import pdb
from torchstat import stat
def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print('模型总大小为：{:.3f}MB'.format(all_size))
        return (param_size, param_sum, buffer_size, buffer_sum, all_size)



class convDU(nn.Module):

    def __init__(self,
                 in_out_channels=2048,
                 kernel_size=(9, 1)
                 ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1,
                      padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, fea):
        n, c, h, w = fea.size()
        # pdb.set_trace()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).reshape(n, c, 1, w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)

        for i in range(h):
            pos = h - i - 1
            if pos == h - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
            # self.conv(fea_stack[pos + 1])
        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea


class convLR(nn.Module):

    def __init__(self,
                 in_out_channels=2048,
                 kernel_size=(1, 9)
                 ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1,
                      padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).reshape(n, c, h, 1)
            # print(i_fea.shape)
            if i == 0:
                fea_stack.append(i_fea)
                continue
                
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)

        for i in range(w):
            pos = w - i - 1
            if pos == w - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]

        fea = torch.cat(fea_stack, 3)
        return fea


class DULRBlock(nn.Module):
    def __init__(self, in_out_channels):
        super(DULRBlock, self).__init__()
        self.in_out_channels = in_out_channels
        self.DULR_layer = nn.Sequential(
            convDU(in_out_channels=in_out_channels, kernel_size=(1, 9)),
            convLR(in_out_channels=in_out_channels, kernel_size=(9, 1)),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_out_channels, in_out_channels, 1)
        )

    def forward(self, x):
        x = self.DULR_layer(x)
        return x

if __name__ == '__main__':
    model=DULRBlock(in_out_channels=128)
    a=torch.randn(4,128,64,64)
    b=model(a)
    print(b.shape)
    getModelSize(model)
    stat(model, (128,64,64))