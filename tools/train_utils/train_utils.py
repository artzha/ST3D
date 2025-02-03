import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

import wandb

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, ft_cfg=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)
        
        # """ BEGIN DEBUG BBOXES """
        # # Save batch as pkl 
        # import pickle
        # batch_np = {}
        # batch_np = {
        #     'points': batch['points'].cpu().numpy(),
        #     'gt_boxes': [gt_box.cpu().numpy() for gt_box in batch['gt_boxes']],
        # }
        # with open('batch.pkl', 'wb') as f:
        #     pickle.dump(batch_np, f)
        # import pdb; pdb.set_trace()
        # import open3d as o3d
        # import numpy as np
        # pc = batch['points'][:, :3].cpu().numpy().astype(np.float32)
        # pc = pc.reshape(-1, 3)
        # pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))

        # # Create a visualizer object
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)  # Set visible=False to run in headless mode
        # vis.add_geometry(pc)
        
        # for i in range(0, batch['gt_boxes'][0].shape[0]):
        #     bbox = batch['gt_boxes'][0][i].cpu().numpy().astype(np.float32)
        #     # create oriented bbox
        #     center = bbox[0:3]
        #     lengths = bbox[3:6]
        #     yaw = bbox[6]
        #     # Get SO3 rotation matrix from yaw
        #     R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
        #     vis.add_geometry(o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=lengths))

        #     # bbox_list.append(o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox)))
        
        # # Set the camera view (optional)
        # view_ctl = vis.get_view_control()
        # view_ctl.set_front([0, 0, -1])
        # view_ctl.set_up([0, -1, 0])
        # view_ctl.set_zoom(0.8)

        # # Update the scene and capture the image
        # vis.poll_events()
        # vis.update_renderer()

        # # Save the rendered image to a file (off-screen)
        # vis.capture_screen_image("output_image.png")

        # # Close the visualizer
        # vis.destroy_window()
        # """ END DEBUG BBOXES """
        
        if ft_cfg is not None:
            wandb.log({"loss": loss, "lr": cur_lr, "iter": cur_it})

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, target_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, ft_cfg=None):
    if ft_cfg is not None:
        # start a new wandb run to track this script
        wandb_name = "lr%0.6f_opt%s_rank%i" % (optim_cfg.LR, optim_cfg.OPTIMIZER, rank)
        model_architecture_name = ft_cfg.ARCH if 'ARCH' in ft_cfg else "PVRCNN"
        wandb.init(
            # set the wandb project where this run will be logged
            project=ft_cfg.WANDB_NAME,
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": optim_cfg.LR,
                "optimizer": optim_cfg.OPTIMIZER,
                "architecture": model_architecture_name,
                "dataset": "CODa",
                "epochs": 25,
                "name": wandb_name
            },
            name=wandb_name
        )
            
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                ft_cfg=ft_cfg
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
