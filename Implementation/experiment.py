from comet_ml import Experiment

import torch
from torchvision import transforms

from src.dataset import AerialLandScape
from src.model import PConvUNet
from src.loss import InpaintingLoss, VGG16FeatureExtractor
from src.train import Trainer
from src.utils import Config, load_ckpt, create_ckpt_dir


# set the config
config = Config("config.yml")
config.ckpt = create_ckpt_dir()
print("Check Point is '{}'".format(config.ckpt))

# Define the used device
device = torch.device("cuda:{}".format(config.cuda_id)
                      if torch.cuda.is_available() else "cpu")

# Define the model
print("Loading the Model...")
model = PConvUNet(finetune=config.finetune,
                  layer_size=config.layer_size)
model.to(device)

img_tf = transforms.Compose([
            transforms.ToTensor()
            ])
mask_tf = transforms.Compose([ transforms.ToTensor() ])

# Define the Validation set
print("Loading the Validation Dataset...")
dataset_val = AerialLandScape(config.data_root,
                      img_tf,
                      mask_tf,
                      data="val")

# Set the configuration for training
if config.mode == "train":
    experiment = None
    print("Loading the Training Dataset...")
    dataset_train = AerialLandScape(config.data_root,
                            img_tf,
                            mask_tf,
                            data="train")
    criterion = InpaintingLoss(VGG16FeatureExtractor(),
                               tv_loss=config.tv_loss).to(device)
    # SET THE LEARNING RATE
    lr = config.initial_lr
    if config.optim == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                     lr=lr,
                                     weight_decay=config.weight_decay)
    elif config.optim == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                    lr=lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

    start_iter = 0
    trainer = Trainer(start_iter, config, device, model, dataset_train,
                      dataset_val, criterion, optimizer, experiment=experiment)
    trainer.iterate(1)

# Set the configuration for testing
elif config.mode == "test":
    pass
    # <model load the trained weights>
    # evaluate(model, dataset_val)
