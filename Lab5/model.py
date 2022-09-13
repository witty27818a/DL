import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.expand = nn.Sequential(
            nn.Linear(24, c_dim),
            nn.ReLU()
        )
        channels = [z_dim + c_dim, 512, 256, 128, 64]
        paddings = [0, 1, 1, 1]
        # strides = [1, 2, 2, 2]
        strides = [2, 2, 2, 2]
        for i in range(1, len(channels)):
            setattr(self, "deconv" + str(i), nn.Sequential(
                nn.ConvTranspose2d(channels[i-1], channels[i], 4, strides[i-1], paddings[i-1]),
                nn.BatchNorm2d(channels[i]),
                # nn.ReLU(inplace = True)
                nn.ReLU()
            ))
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z, c):
        # reshape to (B, z_dim, 1, 1), and then deconvs to get back to a fake image
        z = z.view(-1, self.z_dim, 1, 1)
        c = self.expand(c).view(-1, self.c_dim, 1, 1)
        output = torch.cat((z, c), 1) # (B, z_dim + c_dim, 1, 1)
        output = self.deconv1(output) # (B, 512, 4, 4)
        output = self.deconv2(output) # (B, 256, 8, 8)
        output = self.deconv3(output) # (B, 128, 16, 16)
        output = self.deconv4(output) # (B, 64, 32, 32)
        output = self.deconv5(output) # (B, 3, 64, 64), in [-1, 1]
        return output
    
    def init_weights(self, mean = (0, 1), std = (0.02, 0.02)):
        isinstance(mean, tuple) and isinstance(std, tuple)
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d):
                self._modules[m].weight.data.normal_(mean[0], std[0])
                self._modules[m].bias.data.zero_()
            # elif isinstance(self._modules[m], nn.BatchNorm2d):
            #     self._modules[m].weight.data.normal_(mean[1], std[1])
            #     self._modules[m].bias.data.zero_()

'''看c要不要擴充成H*W*3，然後第一層inchannels變6'''
class Discriminator(nn.Module):
    def __init__(self, img_shape, c_dim):
        super().__init__()
        self.H, self.W, self.C = img_shape
        self.expand = nn.Sequential(
            nn.Linear(24, self.H * self.W),
            nn.LeakyReLU()
        )
        channels = [4, 64, 128, 256, 512]
        for i in range(1, len(channels)):
            setattr(self, "conv" + str(i), nn.Sequential(
                nn.Conv2d(channels[i-1], channels[i], 4, 2, 1),
                nn.BatchNorm2d(channels[i]),
                # nn.LeakyReLU(0.2, inplace = True)
                nn.LeakyReLU()
            ))
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 4),
            nn.Sigmoid()
        )
    
    def forward(self, imgs, c):
        c = self.expand(c).view(-1, 1, self.H, self.W) # c becomes (B, 1, H=64, W=64)
        output = torch.cat((imgs, c), 1) # (B, 4, 64, 64), 4 because 3+1
        output = self.conv1(output) # (B, 64, 32, 32)
        output = self.conv2(output) # (B, 128, 16, 16)
        output = self.conv3(output) # (B, 256, 8, 8)
        output = self.conv4(output) # (B, 512, 4, 4)
        output = self.conv5(output) # (B, 1, 1, 1) output = binary in [0, 1]
        output = torch.squeeze(output) # (B,)
        return output
    
    def init_weights(self, mean = (0, 1), std = (0.02, 0.02)):
        assert isinstance(mean, tuple) and isinstance(std, tuple)
        for m in self._modules:
            if isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean[0], std[0])
                self._modules[m].bias.data.zero_()
            # elif isinstance(self._modules[m], nn.BatchNorm2d):
            #     self._modules[m].weight.data.normal_(mean[1], std[1])
            #     self._modules[m].bias.data.zero_()