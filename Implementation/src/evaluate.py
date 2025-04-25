import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def evaluate(model, dataset, device, filename, experiment=None):
    print('\n⌛ Evaluating the Model')
    model.eval()
    
    # --- Visualization on first 8 images ---
    image_vis, mask_vis, gt_vis = zip(*[dataset[i] for i in range(8)])
    image_vis = torch.stack(image_vis)
    mask_vis = torch.stack(mask_vis)
    gt_vis = torch.stack(gt_vis)
    
    with torch.no_grad():
        output_vis, _ = model(image_vis.to(device), mask_vis.to(device))
    output_vis = output_vis.to(torch.device('cpu'))
    output_comp_vis = mask_vis * image_vis + (1 - mask_vis) * output_vis
    grid = make_grid(torch.cat([image_vis, mask_vis, output_vis, output_comp_vis, gt_vis], dim=0))
    save_image(grid, filename)

    # --- PSNR and SSIM Evaluation ---
    
    psnr_scores = []
    ssim_scores = []

    print("\n🔍 Running PSNR and SSIM evaluation on entire validation set...")

    for i in range(len(dataset)):
        image, mask, gt = dataset[i]
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        with torch.no_grad():
            output, _ = model(image, mask)
        output_comp = mask * image + (1 - mask) * output
        output_comp = output_comp.squeeze().cpu().numpy().transpose(1, 2, 0)
        gt = gt.squeeze().cpu().numpy().transpose(1, 2, 0)

        # Clip for safety (may help with numerical stability)
        output_comp = np.clip(output_comp, 0, 1)
        gt = np.clip(gt, 0, 1)

        psnr = compute_psnr(gt, output_comp, data_range=1.0)
        ssim = compute_ssim(gt, output_comp, channel_axis=2, data_range=1.0)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    mean_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    print(f"📈 Mean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"📈 Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}\n")

    if experiment is not None:
        experiment.log_metric("mean_psnr", mean_psnr)
        experiment.log_metric("std_psnr", std_psnr)
        experiment.log_metric("mean_ssim", mean_ssim)
        experiment.log_metric("std_ssim", std_ssim)

    print('✅ Evaluation Successful!\n')
    
    return mean_psnr, mean_ssim

    
