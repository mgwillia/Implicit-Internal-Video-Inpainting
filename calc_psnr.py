import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

gt_path = '/fs/vulcan-projects/action_augment_hao/gnerv/data/DAVIS/JPEGImages/480p/blackswan'

for epochs in [10, 20, 30, 40, 50]:
    inpaint_path = f'/vulcanscratch/mgwillia/Implicit-Internal-Video-Inpainting/results/blackswan_{epochs}k/'

    images_list = os.listdir(gt_path)
    transform = transforms.Compose([transforms.Resize((320, 600)), transforms.ToTensor()])
    psnr_list = []
    for image_name in images_list:
        gt_image = transform(Image.open(os.path.join(gt_path, image_name)).convert("RGB"))
        inpaint_image = transform(Image.open(os.path.join(inpaint_path, image_name.replace('000', '00'))).convert("RGB"))
        l2_loss = F.mse_loss(inpaint_image, gt_image, reduction='none')
        psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-15)
        psnr_list.append(psnr)

    print(f'PSNR: {torch.stack(psnr_list, 0).cpu().mean().item()}')
