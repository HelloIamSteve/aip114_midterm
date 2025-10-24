import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from config import *
from dataset import *
from model import *
from visualize import *

def get_metric(origin, pred):
    data_range = 255.0
    
    psnr = PSNR(origin, pred, data_range=data_range)
    ssim = SSIM(origin, pred, data_range=data_range, channel_axis=None) # for one channel
    
    return psnr, ssim

@torch.no_grad()
def evaluate_model(model, val_loader, device):
    model.eval()
    
    outputs = []
    total_psnr_gray = 0.0
    total_ssim_gray = 0.0
    total_psnr_rgb = 0.0
    total_ssim_rgb = 0.0
    count = 0

    with torch.no_grad():
        for imgs_noisy, imgs_original in val_loader:
            imgs_noisy = imgs_noisy.to(device)
            imgs_original = imgs_original.to(device)

            # Get model output
            model_outputs = model(imgs_noisy)

            imgs_original = (imgs_original.cpu().numpy().copy() * 255).astype(np.uint8)
            model_outputs = (model_outputs.cpu().numpy() * 255).astype(np.uint8)

            for img_original, output in zip(imgs_original, model_outputs):
                # change order for cv2
                img_original = np.transpose(img_original, (1, 2, 0))
                output = np.transpose(output, (1, 2, 0))
                
                # store
                outputs.append(output)

                # grayscale
                img_original_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
                output_gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
                
                psnr, ssim = get_metric(img_original_gray, output_gray)
                total_psnr_gray += psnr
                total_ssim_gray += ssim
                
                # average of RGB channel
                psnr_rgb_sum = 0.0
                ssim_rgb_sum = 0.0
                
                for i in range(3):
                    original_channel = img_original[:, :, i]
                    output_channel = output[:, :, i]
                    
                    psnr, ssim = get_metric(original_channel, output_channel)
                    psnr_rgb_sum += psnr
                    ssim_rgb_sum += ssim
                
                total_psnr_rgb += psnr_rgb_sum / 3
                total_ssim_rgb += ssim_rgb_sum / 3

                count += 1

    avg_psnr_gray = total_psnr_gray / count
    avg_ssim_gray = total_ssim_gray / count
    avg_psnr_rgb = total_psnr_rgb / count
    avg_ssim_rgb = total_ssim_rgb / count

    return outputs, avg_psnr_gray, avg_ssim_gray, avg_psnr_rgb, avg_ssim_rgb


if __name__ == '__main__':
    # dataset
    midterm100_dataset = Midterm100(root_dir=dataset_dir, mode='val',transform=transforms.ToTensor())

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    models = [DummyModel().to(device),                                      # noisy images
              FilterModel(use_median=False, use_sharpen=False).to(device),  # gaussian
              FilterModel(use_sharpen=False).to(device),                    # median + gaussian
              FilterModel().to(device),                                     # median + gaussian + sharpen
              U_Net().to(device),
              Improved_U_Net(alpha=0.6, beta=0.4).to(device),
              ]
    models[-2].load_model('./U_Net_results/U_Net_MSELoss_best.pt')
    models[-2].name += '_MSELoss'
    models[-1].load_model('./Improved_U_Net_results/Improved_U_Net_MSELoss_best.pt')

    # testing
    outputs = []
    for i, model in enumerate(models):
        loader = DataLoader(midterm100_dataset, batch_size=len(midterm100_dataset))
        model_outputs,\
        avg_psnr_gray, avg_ssim_gray,\
        avg_psnr_mean_rgb, avg_ssim_mean_rgb = evaluate_model(model, loader, device)
        
        outputs.append(model_outputs)

        print(f'{model.name}:')
        print(f"Average PSNR: {avg_psnr_gray:.2f} dB (grayscale), {avg_psnr_mean_rgb:.2f} db (RGB mean)")
        print(f"Average SSIM: {avg_ssim_gray:.4f} (grayscale), {avg_ssim_mean_rgb:.4f} db (RGB mean)")

    # compare model output
    idx = 5
    # idx = 7
    output_compare(midterm100_dataset, models[1: ], outputs[1: ], idx)
    # output_compare(midterm100_dataset, models[1: ], outputs[1: ], idx, crop_range=(0, 128//2, 128//2, 128))
    
    # frequency domain
    freq_error_compare(midterm100_dataset, models, idx, device)