import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, len_Z, hidden_channal, output_channal):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=len_Z,
                out_channels=hidden_channal * 4,
                kernel_size=4,
            ),
            nn.BatchNorm2d(hidden_channal * 4),
            nn.ReLU()
        )
        # [BATCH, hidden_channal * 4 , 4, 4]
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channal * 4,
                out_channels=hidden_channal * 2,
                kernel_size=3,  # 保证生成图像大小为28
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal * 2),
            nn.ReLU()
        )
        #
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channal * 2,
                out_channels=hidden_channal,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channal,
                out_channels=output_channal,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x):
        # [50, 100, 1, 1]
        out = self.layer1(x)
        # [50, 256, 4, 4]
        # print(out.shape)
        out = self.layer2(out)
        # [50, 128, 7, 7]
        # print(out.shape)
        out = self.layer3(out)
        # [50, 64, 14, 14]
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        # [50, 1, 28, 28]
        return out


# # Test Generator
# G = Generator(len_Z, g_hidden_channal, image_channal)
# data = torch.randn((BATCH_SIZE, len_Z, 1, 1))
# print(G(data).shape)


class Discriminator(nn.Module):
    def __init__(self, input_channal, hidden_channal):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channal,
                out_channels=hidden_channal,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channal,
                out_channels=hidden_channal * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal * 2),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channal * 2,
                out_channels=hidden_channal * 4,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channal * 4),
            nn.LeakyReLU(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channal * 4,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0
            ),
            nn.Sigmoid()
        )
        # [BATCH, 1, 1, 1]

    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)

        return out


print(Generator(100, 64, 1))
print(Discriminator(1, 64))
