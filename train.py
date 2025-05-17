# accelerate configuration saved at /home/zhiwei/.cache/huggingface/accelerate/default_config.yaml
# -*- encoding: utf-8 -*-
'''
file       :train.py
Description:
Date       :2025/05/02 13:57:30
Author     :zhiwei tan
version    :python3.9.6
'''
import os
import numpy as np
import argparse
import sys
from tqdm import tqdm
import wandb

import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.optim import AdamW
from lion_pytorch import Lion
import accelerate
from accelerate import Accelerator
import torchvision.transforms as transforms

from data.transform import Simple2DTransform
from data.dataloader import MergedNiiDataset
from util.config import Config
from util import tool

from module.medseg import Unet,MedSegDiff
from module.network import Network,NetworkConfig

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-t', '--task', type=str, default='Brats')

args = parser.parse_args()
config = Config(args.task)

for key, value in config.__dict__.items():
    setattr(args, key, value)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    tool.maybe_mkdir(snapshot_path)
    ckpt_path = os.path.join(snapshot_path, 'ckpts')
    tool.maybe_mkdir(ckpt_path)
    vis_path = os.path.join(snapshot_path, 'vis')
    tool.maybe_mkdir(vis_path)

    logging_dir = os.path.join(snapshot_path, 'log')
    tool.maybe_mkdir(logging_dir)

    # model 

    accelerator = Accelerator(
        gradient_accumulation_steps=args.acc.get('gradient_accumulation_steps'),
        mixed_precision=args.acc.get('mixed_precision'),
        log_with="wandb",
        project_dir =logging_dir
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("med-seg-diff", config=vars(args))

    # ## DEFINE MODEL ##
    # model = Unet(
    #     dim=args.dim,
    #     image_size=args.image_size,
    #     dim_mults=(1, 2, 4, 8),
    #     mask_channels=args.mask_channels,
    #     input_img_channels=args.input_img_channels,
    #     self_condition=args.self_condition
    # )

    network_config = NetworkConfig(
        n_timesteps=args.timesteps,
        n_scales=2, 
        max_patch_size=args.image_size, 
        scale_procedure="loop",
        n_classes=args.num_cls,
        )
    model = Network(network_config)




    # === Dataset ===
    # transform2d = Simple2DTransform(flip_prob=0.5)
    transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
    transform2d = transforms.Compose(transform_list)
    train_set = MergedNiiDataset(
        split="train",
        config=args,
        selected_modalities=['t2','seg'],
        transform=transform2d,
        is_val=False,
        num_cls = args.num_cls,
        suffix ='npy'
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    eval_set = MergedNiiDataset(
        split="eval",
        config=args,
        selected_modalities=['t2','seg'],
        transform=transform2d,
        is_val=True,
        num_cls = args.num_cls,
        suffix ='gz'

    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.acc.get('gradient_accumulation_steps') * args.batch_size * accelerator.num_processes
        )

    ## Initialize optimizer
    if not args.use_lion:
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = Lion(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )
    ## TRAIN MODEL ##
    counter = 0
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps,
        objective='pred_noise'
    ).to(accelerator.device)


    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {args.load_model_from}')


    ## Iterate across training loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        for batch in tqdm(train_loader):
            # if idx%100==0:
            with accelerator.accumulate(model):
                img = batch['image']      # Shape: [B, C, H, W]
                mask = batch['label']      # Shape: [B, num_cls, H, W]
                        
                loss = diffusion(mask, img)
                running_loss += loss.item() * img.size(0)
                accelerator.log({'loss': loss})  # Log loss to wandb
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        counter += 1
        epoch_loss = running_loss / len(train_loader)
        print('Training Loss : {:.4f}'.format(epoch_loss))
        # ## INFERENCE ##
        # if epoch % args.freq.get('val_epoch', 10) == 0:
        #     diffusion.model.eval()
        #     total_dice, total_iou = [], []

        #     with torch.no_grad():
        #         for idx, eval_batch in enumerate(eval_loader):
        #             if idx%500 == 20:

        #                 img = eval_batch['image'].to(accelerator.device)   # [B, C, H, W]
        #                 mask = eval_batch['label'].to(accelerator.device)  # [B, num_cls, H, W]

        #                 pred = diffusion.sample(img)  # [B, num_cls, H, W]
        #                 pred = torch.softmax(pred, dim=1)

        #                 # get predicted class
        #                 pred_cls = torch.argmax(pred, dim=1)     # [B, H, W]
        #                 mask_cls = torch.argmax(mask, dim=1)     # [B, H, W]
                        
        #                 # # Debug check
        #                 # print("Unique in pred_cls:", torch.unique(pred_cls[0]))
        #                 # print("Unique in mask_cls:", torch.unique(mask_cls[0]))
        #                 # assert torch.unique(pred_cls[0])==torch.unique(mask_cls[0])

        #                 # one-hot for Dice
        #                 pred_onehot = F.one_hot(pred_cls, num_classes=mask.shape[1]).permute(0, 3, 1, 2).float()
        #                 mean_dice, _ = tool.multiclass_dice(pred_onehot, mask)
        #                 total_dice.append(mean_dice)

        #                 mean_iou, _ = tool.multiclass_iou(pred_cls.cpu(), mask_cls.cpu(), num_classes=mask.shape[1])
        #                 total_iou.append(mean_iou)


        #             # --- Logging first example ---
        #             if idx%500 == 20:
        #                 input_img = img[0, 0].cpu().numpy()
        #                 pred_color = tool.colorize_label(pred_cls[0], num_classes=mask.shape[1])
        #                 gt_color = tool.colorize_label(mask_cls[0], num_classes=mask.shape[1])

        #                 for tracker in accelerator.trackers:
        #                     if tracker.name == "wandb":
        #                         tracker.log({
        #                             'val/pred-img-mask/epoch_{epoch}_batch_{idx}': [
        #                                 wandb.Image(input_img, caption=f'Epoch {epoch}: Input'),
        #                                 wandb.Image(pred_color, caption=f'Epoch {epoch}: Prediction'),
        #                                 wandb.Image(gt_color, caption=f'Epoch {epoch}: Ground Truth'),
        #                             ]
        #                         })
        #     # --- summary ---
        #     mean_dice_all = np.mean(total_dice)
        #     mean_iou_all = np.mean(total_iou)
        #     print(f"Epoch {epoch} — Mean Dice: {mean_dice_all:.4f}, Mean IoU: {mean_iou_all:.4f}")
        #     accelerator.log({
        #         'val/mean_dice': mean_dice_all,
        #         'val/mean_iou': mean_iou_all
        #     }, step=epoch)
        #     diffusion.model.train() 
            
            
                                                                                  
        if epoch % args.freq.get('save_epoch', 30) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(ckpt_path, f'state_dict_epoch_{epoch}_loss_{epoch_loss:.4f}.pt'))
            with torch.no_grad():
                pred = diffusion.sample(img)                     # [B, C, H, W]
                pred = torch.softmax(pred, dim=1)                # softmax over classes
                pred_cls = torch.argmax(pred, dim=1)             # [B, H, W]

                mask_cls = torch.argmax(mask, dim=1)             # [B, H, W] from one-hot

                input_img = img[0, 0].cpu().numpy()
                pred_color = tool.colorize_label(pred_cls[0], num_classes=mask.shape[1])
                gt_color = tool.colorize_label(mask_cls[0], num_classes=mask.shape[1])

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log({
                            f'val/pred-img-mask/epoch_{epoch}': [
                                wandb.Image(input_img, caption=f'Epoch {epoch}: Input'),
                                wandb.Image(pred_color, caption=f'Epoch {epoch}: Prediction'),
                                wandb.Image(gt_color, caption=f'Epoch {epoch}: Ground Truth'),
                            ]
                        })



if __name__=="__main__":
    main()
