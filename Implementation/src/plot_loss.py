import matplotlib.pyplot as plt
import os

def plot_loss_graphs(loss_history, epoch_number, save_path):
  print('\n⌛ Saving Loss History plot over Epoch: {epoch_number}')
  epoch_losses = loss_history[epoch_number]
  loss_keys = list(next(iter(epoch_losses.values())).keys())
  step_numbers = sorted(epoch_losses.keys())

  plt.figure(figsize=(12, 6))

  for loss_key in loss_keys:
      loss_values = [epoch_losses[step][loss_key] for step in step_numbers]
      plt.plot(step_numbers, loss_values, label=loss_key)

  plt.xlabel("Step Number")
  plt.ylabel("Loss Value")
  plt.title(f"Losses over Steps (Epoch {epoch_number})")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()

  os.makedirs(save_path, exist_ok=True)
  filename = os.path.join(save_path, f"epoch_{epoch_number}_loss_plot.png")
  plt.savefig(filename)
  plt.close()

  print('✅ Loss History plot over Epoch: {epoch_number} successfully saved!\n')
  
def plot_metric_graphs(psnr_list, ssim_list, step_interval, save_dir):
  os.makedirs(save_dir, exist_ok=True)
  steps = [i * step_interval for i in range(len(psnr_list))]

  # Plot PSNR
  plt.figure(figsize=(8, 5))
  plt.plot(steps, psnr_list, marker='o', color='blue', label='PSNR')
  plt.title('PSNR Over Time')
  plt.xlabel('Step')
  plt.ylabel('PSNR (dB)')
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, 'psnr_history.png'))
  plt.close()

  # Plot SSIM
  plt.figure(figsize=(8, 5))
  plt.plot(steps, ssim_list, marker='o', color='green', label='SSIM')
  plt.title('SSIM Over Time')
  plt.xlabel('Step')
  plt.ylabel('SSIM')
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, 'ssim_history.png'))
  plt.close()

