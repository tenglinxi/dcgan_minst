import torch
from model import  Generator,Discriminator
from torch.autograd import Variable
from data import dataloader

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
EPOCH = 10  # 训练整批数据多少次
LR = 0.0002  # 学习率
DOWNLOAD_MNIST = False  # 已经下载好的话，会自动跳过的
len_Z = 100  # random input.channal for Generator
g_hidden_channal = 64
d_hidden_channal = 64
image_channal = 1  # mnist数据为黑白的只有一维
generator = Generator(len_Z, g_hidden_channal, image_channal)
discriminator = Discriminator(image_channal, g_hidden_channal)
BATCH_SIZE = 32
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
import torchvision





# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
import matplotlib.pyplot as plt
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
from torchvision.utils import save_image
import numpy as np
for epoch in range(200):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100, 1, 1))))

        # Generate a batch of images

        gen_imgs = generator(z)

        print(discriminator(gen_imgs).size(), valid.size())

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, 200, i, len(dataloader), d_loss.item(), g_loss.item())
        )



        batches_done = epoch * len(dataloader) + i
        if batches_done % 20 == 0:
            save_image(gen_imgs.data[:4], "images/%d.png" % batches_done, nrow=5, normalize=True)
    picture = torch.squeeze(gen_imgs[0].cpu()).detach().numpy()
    plt.imshow(picture, cmap=plt.cm.gray_r)
    plt.show()