# train.py
import os
import copy
import logging
import sys
import yaml
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch
# from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.bing_rgb import make_bingrgb  # Import the new dataset loader
from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):

    # wandb.login(key="your_wandb_key")

    # wandb.init(
    #     project="ijepa-training",
    #     entity="your_entity",
    #     id="your_id",
    #     resume="allow",
    #     config=args
    # )

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    # _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
    #         transform=transform,
    #         batch_size=batch_size,
    #         collator=mask_collator,
    #         pin_mem=pin_mem,
    #         training=True,
    #         num_workers=num_workers,
    #         world_size=world_size,
    #         rank=rank,
    #         root_path=root_path,
    #         image_folder=image_folder,
    #         copy_data=copy_data,
    #         drop_last=True)
    dataset, unsupervised_loader = make_bingrgb(
        image_file=os.path.join(root_path, 'train', 'dhaka_train.tif'),
        label_file=os.path.join(root_path, 'train', 'dhaka_train_gt.tif'),
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        drop_last=True)

    logger.info('Created ImageNet dataloaders.')

    # -- init optimizer
    param_groups = [
        {'params': encoder.parameters()},
        {'params': predictor.parameters()}]
    opt, scheduler, steps_per_epoch = init_opt(param_groups, batch_size,
                                               start_lr, lr, final_lr,
                                               num_epochs, warmup,
                                               wd, final_wd, ipe_scale,
                                               unsupervised_loader)

    # -- init EMA
    if ema:
        logger.info('Using EMA.')
        for param in target_encoder.parameters():
            param.detach_()
        source_params = list(encoder.parameters())
        target_params = list(target_encoder.parameters())
        alpha = 0.996
        one_minus_alpha = 1. - alpha

    # -- init AMP
    if use_bfloat16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # -- init model to device
    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    if load_model:
        ckpt = load_checkpoint(load_path, device)
        encoder.load_state_dict(ckpt['encoder'])
        predictor.load_state_dict(ckpt['predictor'])
        target_encoder.load_state_dict(ckpt['target_encoder'])
        opt.load_state_dict(ckpt['opt'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        logger.info(f'Read checkpoint {load_path} epoch {start_epoch}')
    else:
        start_epoch = 0
        logger.info('No checkpoint, training from scratch.')

    encoder = DistributedDataParallel(encoder, device_ids=[device])
    predictor = DistributedDataParallel(predictor, device_ids=[device])

    logger.info('Starting main training loop')
    for epoch in range(start_epoch, num_epochs):

        encoder.train()
        predictor.train()
        if ema:
            target_encoder.train()

        unsupervised_loader.sampler.set_epoch(epoch)

        unsupervised_loader_iter = iter(unsupervised_loader)

        if rank == 0:
            pbar = tqdm(total=steps_per_epoch, desc=f'Epoch {epoch}')
        else:
            pbar = None

        for i in range(steps_per_epoch):
            data, mask = next(unsupervised_loader_iter)
            data = data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(scaler is not None):
                emb_enc = encoder(data)
                emb_pred = predictor(apply_masks(emb_enc, mask))
                loss = F.mse_loss(emb_enc, emb_pred)
                assert not torch.isnan(loss), 'Loss is NaN.'

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            scheduler.step()

            if ema:
                for j in range(len(source_params)):
                    target_params[j].mul_(alpha)
                    target_params[j].add_(source_params[j].detach().mul_(one_minus_alpha))

            if rank == 0:
                pbar.update(1)
                if i % log_freq == 0:
                    l_maskA, l_maskB = mask_collator.loss_terms()
                    time_ms = gpu_timer(False)
                    csv_logger.log(epoch, i, loss.item(), l_maskA, l_maskB, time_ms)
                    log_info = {
                        "epoch": epoch,
                        "iteration": i,
                        "loss": loss.item(),
                        "mask_A": l_maskA,
                        "mask_B": l_maskB,
                        "time_ms": time_ms
                    }
                    # wandb.log(log_info)

        if rank == 0:
            pbar.close()
            save_dict = {
                'encoder': encoder.module.state_dict(),
                'predictor': predictor.module.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'opt': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_dict, latest_path)
            if (epoch % checkpoint_freq == 0) or ((epoch+1) == num_epochs):
                torch.save(save_dict, save_path)

    csv_logger.close()
    logger.info('Training completed successfully.')
