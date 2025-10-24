import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Midterm100(torch.utils.data.Dataset):
    # im_82 is deleted
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.noisy_dir = os.path.join(root_dir, 'noisy')
        self.original_dir = os.path.join(root_dir, 'original')
        self.mode = mode
        self.transform = transform

        # get filenames and sort them
        noisy_filenames = sorted(os.listdir(self.noisy_dir),
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))
        original_filenames = sorted(os.listdir(self.original_dir),
                                    key=lambda x: int(x.split('_')[1].split('.')[0]))

        noisy_filenames = [f for f in noisy_filenames if f.endswith('.png')]
        original_filenames = [f for f in original_filenames if f.endswith('.png')]

        # use mode to split dataset
        if self.mode == 'train':    # 1~80
            self.noisy_filenames = noisy_filenames[:80]
            self.original_filenames = original_filenames[:80]
        else:                       # 81~100
            self.noisy_filenames = noisy_filenames[80:]
            self.original_filenames = original_filenames[80:]

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, idx):
        # read image and return in rgb order
        img_noisy_name = os.path.join(self.noisy_dir, self.noisy_filenames[idx])
        img_noisy = cv2.imread(img_noisy_name)
        img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB)
        
        img_original_name = os.path.join(self.original_dir, self.original_filenames[idx])
        img_original = cv2.imread(img_original_name)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_noisy = self.transform(img_noisy)
            img_original = self.transform(img_original)
        
        return img_noisy, img_original

class CBSD68(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', train_size=50, noisy=15,transform=None):
        self.root_dir = root_dir
        self.noisy = noisy
        self.noisy_dir = os.path.join(root_dir, f'noisy{self.noisy}')
        self.original_dir = os.path.join(root_dir, 'original_png')
        self.mode = mode
        self.transform = transform

        noisy_filenames = os.listdir(self.noisy_dir)
        original_filenames = os.listdir(self.original_dir)
        
        if self.mode == 'train':
            self.noisy_filenames = noisy_filenames[:train_size]
            self.original_filenames = original_filenames[:train_size]
        elif self.mode == 'val':
            self.noisy_filenames = noisy_filenames[train_size:]
            self.original_filenames = original_filenames[train_size:]
        else:
            raise ValueError('Mode must be "train" or "val"')

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, idx):
        # read image and return in rgb order
        noisy_img_name = os.path.join(self.noisy_dir, self.noisy_filenames[idx])
        noisy_img = cv2.imread(noisy_img_name)
        noisy_img = cv2.resize(noisy_img, (noisy_img.shape[1]-1, noisy_img.shape[0]-1))
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        
        original_img_name = os.path.join(self.original_dir, self.original_filenames[idx])
        original_img = cv2.imread(original_img_name)
        original_img = cv2.resize(original_img, (original_img.shape[1]-1, original_img.shape[0]-1))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            noisy_img = self.transform(noisy_img)
            original_img = self.transform(original_img)
        
        if noisy_img.shape != (3, 480, 320):
            noisy_img = torch.rot90(noisy_img, dims=(1, 2))
            original_img = torch.rot90(original_img, dims=(1, 2))

        return noisy_img, original_img

if __name__ == '__main__':
    dataset_dir = os.path.join('Midterm100', 'Midterm100')
    midterm100_dataset = Midterm100(root_dir=dataset_dir, transform=transforms.ToTensor())
    print(f'Dataset size: {len(midterm100_dataset)}')

    test_idx = 3
    noisy_img, original_img = midterm100_dataset[test_idx]
    print(noisy_img.shape)

    # dataset_dir = os.path.join('CBSD68-dataset', 'CBSD68')
    # cbsd68_dataset = CBSD68(root_dir=dataset_dir, transform=transforms.ToTensor())
    # print(f'Dataset size: {len(cbsd68_dataset)}')

    # test_idx = 3
    # noisy_img, original_img = cbsd68_dataset[test_idx]
    # print(noisy_img.shape)

    noisy_img = (noisy_img.permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
    original_img = (original_img.permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
    plt.imsave('sample_noisy.png', noisy_img)
    plt.imsave('sample_original.png', original_img)    