import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Original noisy'

    def forward(self, x):
        return x
    
class FilterModel(nn.Module):
    def __init__(self, gaussian_kernel=3, gaussian_sigma=1, median_kernel=3,\
                 use_median=True, use_sharpen=True, sharpen_alpha=3):
        super().__init__()

        # Gaussian filter
        self.gaussian_kernel = gaussian_kernel
        self.gaussian = transforms.GaussianBlur(kernel_size=gaussian_kernel, sigma=gaussian_sigma)
        self.name = 'Gaussian filter'

        # Median filter
        self.median_kernel = median_kernel
        self.use_median = use_median
        if use_median:
            self.name = 'Median + gaussian filter'

        # sharpen
        self.use_sharpen = use_sharpen
        self.sharpen_alpha = sharpen_alpha
        if use_sharpen:
            self.name = 'Median + gaussian filter + sharpen'

    def forward(self, x):
        y = self.gaussian(x)

        if self.use_median:
            batch_output = []
            for img_tensor in x:
                # Convert to NumPy format for OpenCV
                img_np_chw = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_np_hwc = np.transpose(img_np_chw, (1, 2, 0))
                
                # Apply cv2.medianBlur
                filtered_np_hwc = cv2.medianBlur(img_np_hwc, self.median_kernel)
                
                # Convert back to PyTorch Tensor format
                filtered_np_chw = np.transpose(filtered_np_hwc, (2, 0, 1))
                filtered_tensor = torch.from_numpy(filtered_np_chw).float() / 255.0
                batch_output.append(filtered_tensor)

            x_med = torch.stack(batch_output).to(x.device)
            y = self.gaussian(x_med)

        if self.use_sharpen:
            batch_output = []
            for img_tensor in y:
                # Convert to NumPy format for OpenCV, keeping float32 for calculations
                img_np_chw = img_tensor.cpu().numpy()
                img_np_hwc = np.transpose(img_np_chw, (1, 2, 0))

                # sharpening
                img_hbf_float = img_np_hwc * 255.0
                img_mask = img_hbf_float - cv2.GaussianBlur(img_hbf_float, (self.gaussian_kernel, self.gaussian_kernel), 0)
                img_sharpened = cv2.addWeighted(img_hbf_float, 1, img_mask, self.sharpen_alpha, 0)
                img_sharpened_clipped = np.clip(img_sharpened, 0, 255)

                # Convert back to PyTorch Tensor format
                img_back_chw = np.transpose(img_sharpened_clipped, (2, 0, 1))
                filtered_tensor = torch.from_numpy(img_back_chw).float() / 255.0
                batch_output.append(filtered_tensor)
                y = torch.stack(batch_output).to(y.device)

        return y
    
class U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'U_Net'
        
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # input 3 channels for RGB image
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)    # upsampling
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # Input is 128 (from upconv) + 128 (from skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # Input is 64 (from upconv) + 64 (from skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final Output Layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid() # to map to [0, 1]

    def forward(self, x):
        # Encoder
        skip1 = self.enc_conv1(x)
        x = self.pool1(skip1)
        
        skip2 = self.enc_conv2(x)
        x = self.pool2(skip2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv1(x)
        # Skip Connection 1
        x = torch.cat((skip2, x), dim=1) 
        x = self.dec_conv1(x)

        x = self.upconv2(x)
        x = torch.cat((skip1, x), dim=1)
        x = self.dec_conv2(x)

        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x

    def save_model(self, path, record=True):
        try:
            if record:
                print(f'save model at {path}')
            torch.save(self.state_dict(), path)
        except Exception as e:
            print(e)
        
    def load_model(self, path):
        try:
            print(f'load model from {path}')
            self.load_state_dict(torch.load(path))
        except Exception as e:
            print(e)

class DWTPCA_Downsample(nn.Module):
    def __init__(self, in_channels, wave='haar', alpha=0.5, beta=0.5):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)

        # use Conv2d to downsample
        self.pca_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        # DWT produces a low-pass (LL) and high-pass (LH, HL, HH) components
        ll, _ = self.dwt(x)

        pca_out = self.pca_conv(x)
        fused_output = (self.alpha * ll) + (self.beta * pca_out)
        return fused_output

class IDWT_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.idwt = DWTInverse(mode='zero', wave=wave)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        high_freq_shape = list(x.shape)
        high_freq_shape.insert(2, 3) # shape -> [N, C, 3, H, W]
        high_freq_zeros = torch.zeros(high_freq_shape, device=x.device, dtype=x.dtype)
        
        upsampled = self.idwt((x, [high_freq_zeros]))
        
        output = self.conv(upsampled)
        return output

class Improved_U_Net(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.name = 'Improved_U_Net'
        
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Replaced MaxPool2d with DWTPCA_Downsample
        self.down1 = DWTPCA_Downsample(64, alpha=alpha, beta=beta)
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Replaced MaxPool2d with DWTPCA_Downsample
        self.down2 = DWTPCA_Downsample(128, alpha=alpha, beta=beta)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        # Replaced ConvTranspose2d with IDWT_Upsample
        self.up1 = IDWT_Upsample(256, 128)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # Input: 128 (from upsample) + 128 (from skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Replaced ConvTranspose2d with IDWT_Upsample
        self.up2 = IDWT_Upsample(128, 64)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # Input: 64 (from upsample) + 64 (from skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final Output Layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        skip1 = self.enc_conv1(x)
        x = self.down1(skip1)
        
        skip2 = self.enc_conv2(x)
        x = self.down2(skip2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = torch.cat((skip2, x), dim=1) 
        x = self.dec_conv1(x)

        x = self.up2(x)
        x = torch.cat((skip1, x), dim=1)
        x = self.dec_conv2(x)

        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x

    def save_model(self, path, record=True):
        try:
            if record:
                print(f'save model at {path}')
            torch.save(self.state_dict(), path)
        except Exception as e:
            print(e)
            
    def load_model(self, path):
        try:
            print(f'load model from {path}')
            self.load_state_dict(torch.load(path))
        except Exception as e:
            print(e)
