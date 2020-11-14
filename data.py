import torchvision
import torch.utils.data as Data

BATCH_SIZE = 32
class myMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, targetNum=None):
        super(myMNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)
        if targetNum:
            self.data = self.data[self.targets == targetNum]

            self.data = self.data[:int(self.__len__() / BATCH_SIZE) * BATCH_SIZE]

            self.targets = self.targets[self.targets == targetNum][
                                :int(self.__len__() / BATCH_SIZE) * BATCH_SIZE]

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return 10000

train_data = myMNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=True,  # 没下载就下载, 下载了就不用再下了
    targetNum=1
)

dataloader =Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱顺序
)
