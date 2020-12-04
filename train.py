# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import sys
from os import path as osp
import time
import datetime
import simplejson as json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

import numpy as np

from tensorboardX import SummaryWriter

import forge
from forge import flags
import forge.experiment_tools as fet
from forge.experiment_tools import fprint

from utils.geco import GECO
from scripts.compute_fid import fid_from_model


# ELBO divergence threshold for stopping training
ELBO_DIV = 1e8


def main_flags():
    # Data & model config
    flags.DEFINE_string('data_config', 'datasets/multid_config.py',
                        'Path to a data config file.')
    flags.DEFINE_string('model_config', 'models/genesis_config.py',
                        'Path to a model config file.')
    # Logging config
    flags.DEFINE_string('results_dir', 'checkpoints',
                        'Top directory for all experimental results.')
    flags.DEFINE_string('run_name', 'test',
                        'Name of this job and name of results folder.')
    flags.DEFINE_integer('report_loss_every', 1000,
                         'Number of iterations between reporting minibatch loss.')
    flags.DEFINE_integer('run_validation_every', 25000,
                         'How many equally spaced validation runs to do.')
    flags.DEFINE_integer('num_checkpoints', 4,
                         'How many equally spaced model checkpoints to save.')
    flags.DEFINE_boolean('resume', False, 'Tries to resume a job if True.')
    flags.DEFINE_boolean('log_grads_and_weights', False,
                         'Log gradient and weight histograms - storage intensive!')
    flags.DEFINE_boolean('log_distributions', False,
                         'Log mu and sigma of posterior and prior distributions.')
    # Optimisation config
    flags.DEFINE_integer('train_iter', 500000,
                         'Number of training iterations.')
    flags.DEFINE_integer('batch_size', 32, 'Mini-batch size.')
    flags.DEFINE_string('optimiser', 'adam', 'Optimiser for updating weights.')
    flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
    flags.DEFINE_integer('N_eval', 10000,
                         'Number of samples to run evaluation on.')
    # Loss config
    flags.DEFINE_float('beta', 0.5, 'KL weighting.')
    flags.DEFINE_boolean('beta_warmup', True, 'Warm up beta.')
    flags.DEFINE_boolean('geco', True, 'Use GECO objective.')
    flags.DEFINE_float('g_goal', 0.5655, 'GECO recon goal.')
    flags.DEFINE_float('g_lr', 1e-5, 'GECO learning rate.')
    flags.DEFINE_float('g_alpha', 0.99, 'GECO momentum for error.')
    flags.DEFINE_float('g_init', 1.0, 'GECO inital Lagrange factor.')
    flags.DEFINE_float('g_min', 1e-10, 'GECO min Lagrange factor.')
    flags.DEFINE_float('g_speedup', 10., 'Scale GECO lr if delta positive.')
    # Other
    flags.DEFINE_boolean('gpu', True, 'Use GPU if available.')
    flags.DEFINE_boolean('multi_gpu', False, 'Use multiple GPUs if available.')
    flags.DEFINE_boolean('debug', False, 'Debug flag.')
    flags.DEFINE_integer('seed', 0, 'Seed for random number generators.')


def main():
    # ------------------------
    # SETUP
    # ------------------------

    # Parse flags
    config = forge.config()
    if config.debug:
        config.num_workers = 0
        config.batch_size = 2

    # Fix seeds. Always first thing to be done after parsing the config!
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup checkpoint or resume
    logdir = osp.join(config.results_dir, config.run_name)
    if config.resume and not osp.exists(logdir):
        fprint((f"WARNING: Tried to resume training but couldn't find logdir; "
               "training form scratch instead"), True)
        config.resume = False
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume)
    checkpoint_name = osp.join(logdir, 'model.ckpt')

    # Using GPU(S)?
    if torch.cuda.is_available() and config.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        config.gpu = False
        torch.set_default_tensor_type('torch.FloatTensor')
    fprint(f"Use GPU: {config.gpu}")
    if config.gpu and config.multi_gpu and torch.cuda.device_count() > 1:
        fprint(f"Using {torch.cuda.device_count()} GPUs!")
        config.num_workers = torch.cuda.device_count() * config.num_workers
    else:
        config.multi_gpu = False

    # Print flags
    # fet.print_flags()
    # TODO(martin) make this cleaner
    fprint(json.dumps(fet._flags.FLAGS.__flags, indent=4, sort_keys=True))

    # Setup TensorboardX SummaryWriter
    writer = SummaryWriter(logdir)

    # Load data
    train_loader, val_loader, test_loader = fet.load(config.data_config, config)
    num_elements = 3 * config.img_size**2  # Assume three input channels

    # Load model
    model = fet.load(config.model_config, config)
    fprint(model)
    if config.geco:
        # Goal is specified per pixel & channel so it doesn't need to
        # be changed for different resolutions etc.
        geco_goal = config.g_goal * num_elements
        # Scale step size to get similar update at different resolutions
        geco_lr = config.g_lr * (64**2 / config.img_size**2)
        geco = GECO(geco_goal, geco_lr, config.g_alpha, config.g_init,
                    config.g_min, config.g_speedup)
        beta = geco.beta
    else:
        beta = torch.tensor(config.beta)

    # Setup optimiser
    if config.optimiser == 'rmsprop':
        optimiser = optim.RMSprop(model.parameters(), config.learning_rate)
    elif config.optimiser == 'adam':
        optimiser = optim.Adam(model.parameters(), config.learning_rate)
    elif config.optimiser == 'sgd':
        optimiser = optim.SGD(model.parameters(), config.learning_rate, 0.9)

    # Try to restore model and optimiser from checkpoint
    iter_idx = 0
    if resume_checkpoint is not None:
        fprint(f"Restoring checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        # Restore model & optimiser
        model_state_dict = checkpoint['model_state_dict']
        model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_1', None)
        model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_2', None)
        model.load_state_dict(model_state_dict)
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        # Restore GECO
        if config.geco and 'beta' in checkpoint:
            geco.beta = checkpoint['beta']
        if config.geco and 'err_ema' in checkpoint:
            geco.err_ema = checkpoint['err_ema']
        # Update starting iter
        iter_idx = checkpoint['iter_idx'] + 1
    fprint(f"Starting training at iter = {iter_idx}")

    # Push model to GPU(s)?
    if config.multi_gpu:
        fprint("Wrapping model in DataParallel.")
        model = nn.DataParallel(model)
    if config.gpu:
        fprint("Pushing model to GPU.")
        model = model.cuda()
        if config.geco:
            geco.to_cuda()

    # ------------------------
    # TRAINING
    # ------------------------

    model.train()
    timer = time.time()
    while iter_idx <= config.train_iter:
        for train_batch in train_loader:
            # Parse data
            train_input = train_batch['input']
            if config.gpu:
                train_input = train_input.cuda()

            # Forward propagation
            optimiser.zero_grad()
            output, losses, stats, att_stats, comp_stats = model(train_input)

            # Reconstruction error
            err = losses.err.mean(0)
            # KL divergences
            kl_m, kl_l = torch.tensor(0), torch.tensor(0)
            # -- KL stage 1
            if 'kl_m' in losses:
                kl_m = losses.kl_m.mean(0)
            elif 'kl_m_k' in losses:
                kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
            # -- KL stage 2
            if 'kl_l' in losses:
                kl_l = losses.kl_l.mean(0)
            elif 'kl_l_k' in losses:
                kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()

            # Compute ELBO
            elbo = (err + kl_l + kl_m).detach()
            err_new = err.detach()
            kl_new = (kl_m + kl_l).detach()
            # Compute MSE / RMSE
            mse_batched = ((train_input-output)**2).mean((1, 2, 3)).detach()
            rmse_batched = mse_batched.sqrt()
            mse, rmse = mse_batched.mean(0), rmse_batched.mean(0)

            # Main objective
            if config.geco:
                loss = geco.loss(err, kl_l + kl_m)
                beta = geco.beta
            else:
                if config.beta_warmup:
                    # Increase beta linearly over 20% of training
                    beta = config.beta*iter_idx / (0.2*config.train_iter)
                    beta = torch.tensor(beta).clamp(0, config.beta)
                else:
                    beta = config.beta
                loss = err + beta*(kl_l + kl_m)

            # Backprop and optimise
            loss.backward()
            optimiser.step()

            # Heartbeat log
            if (iter_idx % config.report_loss_every == 0 or
                    float(elbo) > ELBO_DIV or config.debug):
                # Print output and write to file
                ps = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ps += f' {config.run_name} | '
                ps += f'[{iter_idx}/{config.train_iter:.0e}]'
                ps += f' elb: {float(elbo):.0f} err: {float(err):.0f} '
                if 'kl_m' in losses or 'kl_m_k' in losses:
                    ps += f' klm: {float(kl_m):.1f}'
                if 'kl_l' in losses or 'kl_l_k' in losses:
                    ps += f' kll: {float(kl_l):.1f}'
                ps += f' bet: {float(beta):.1e}'
                s_per_b = (time.time()-timer)
                if not config.debug:
                    s_per_b /= config.report_loss_every
                timer = time.time()  # Reset timer
                ps += f' - {s_per_b:.2f} s/b'
                fprint(ps)

                # TensorBoard logging
                # -- Optimisation stats
                writer.add_scalar('optim/beta', beta, iter_idx)
                writer.add_scalar('optim/s_per_batch', s_per_b, iter_idx)
                if config.geco:
                    writer.add_scalar('optim/geco_err_ema',
                                      geco.err_ema, iter_idx)
                    writer.add_scalar('optim/geco_err_ema_element',
                                      geco.err_ema/num_elements, iter_idx)
                # -- Main loss terms
                writer.add_scalar('train/err', err, iter_idx)
                writer.add_scalar('train/err_element', err/num_elements, iter_idx)
                writer.add_scalar('train/kl_m', kl_m, iter_idx)
                writer.add_scalar('train/kl_l', kl_l, iter_idx)
                writer.add_scalar('train/elbo', elbo, iter_idx)
                writer.add_scalar('train/loss', loss, iter_idx)
                writer.add_scalar('train/mse', mse, iter_idx)
                writer.add_scalar('train/rmse', rmse, iter_idx)
                # -- Per step loss terms
                for key in ['kl_l_k', 'kl_m_k']:
                    if key not in losses: continue
                    for step, val in enumerate(losses[key]):
                        writer.add_scalar(f'train_steps/{key}{step}',
                                          val.mean(0), iter_idx)
                # -- Attention stats
                if config.log_distributions and att_stats is not None:
                    for key in ['mu_k', 'sigma_k', 'pmu_k', 'psigma_k']:
                        if key not in att_stats: continue
                        for step, val in enumerate(att_stats[key]):
                            writer.add_histogram(f'att_{key}_{step}',
                                                 val, iter_idx)
                # -- Component stats
                if config.log_distributions and comp_stats is not None:
                    for key in ['mu_k', 'sigma_k', 'pmu_k', 'psigma_k']:
                        if key not in comp_stats: continue
                        for step, val in enumerate(comp_stats[key]):
                            writer.add_histogram(f'comp_{key}_{step}',
                                                 val, iter_idx)

            # Save checkpoints
            ckpt_freq = config.train_iter / config.num_checkpoints
            if iter_idx % ckpt_freq == 0:
                ckpt_file = '{}-{}'.format(checkpoint_name, iter_idx)
                fprint(f"Saving model training checkpoint to: {ckpt_file}")
                if config.multi_gpu:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                ckpt_dict = {'iter_idx': iter_idx,
                             'model_state_dict': model_state_dict,
                             'optimiser_state_dict': optimiser.state_dict(),
                             'elbo': elbo}
                if config.geco:
                    ckpt_dict['beta'] = geco.beta
                    ckpt_dict['err_ema'] = geco.err_ema
                torch.save(ckpt_dict, ckpt_file)

            # Run validation and log images
            if (iter_idx % config.run_validation_every == 0 or
                    float(elbo) > ELBO_DIV):
                # Weight and gradient histograms
                if config.log_grads_and_weights:
                    for name, param in model.named_parameters():
                        writer.add_histogram(f'weights/{name}', param.data,
                                             iter_idx)
                        writer.add_histogram(f'grads/{name}', param.grad,
                                             iter_idx)
                # TensorboardX logging - images
                visualise_inference(model, train_batch, writer, 'train', 
                                    iter_idx)
                # Validation
                fprint("Running validation...")
                eval_model = model.module if config.multi_gpu else model
                evaluation(eval_model, val_loader, writer, config, iter_idx,
                           N_eval=config.N_eval)

            # Increment counter
            iter_idx += 1
            if iter_idx > config.train_iter:
                break

            # Exit if training has diverged
            if elbo.item() > ELBO_DIV:
                fprint(f"ELBO: {elbo.item()}")
                fprint(f"ELBO has exceeded {ELBO_DIV} - training has diverged.")
                sys.exit()

    # ------------------------
    # TESTING
    # ------------------------

    # Save final checkpoint
    ckpt_file = '{}-{}'.format(checkpoint_name, 'FINAL')
    fprint(f"Saving model training checkpoint to: {ckpt_file}")
    if config.multi_gpu:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    ckpt_dict = {'iter_idx': iter_idx,
                 'model_state_dict': model_state_dict,
                 'optimiser_state_dict': optimiser.state_dict()}
    if config.geco:
        ckpt_dict['beta'] = geco.beta
        ckpt_dict['err_ema'] = geco.err_ema
    torch.save(ckpt_dict, ckpt_file)

    # Test evaluation
    fprint("STARTING TESTING...")
    eval_model = model.module if config.gpu and config.multi_gpu else model
    final_elbo = evaluation(
        eval_model, test_loader, None, config, iter_idx, N_eval=config.N_eval)
    fprint(f"TEST ELBO = {float(final_elbo)}")

    # FID computation
    try:
        fid_from_model(model, test_loader, img_dir=osp.join('/tmp', logdir))
    except NotImplementedError:
        fprint("Sampling not implemented for this model.")

    # Close writer
    writer.close()


def visualise_inference(model, vis_batch, writer, mode, iter_idx):
    # Only visualise for eight images
    # Forward pass
    vis_input = vis_batch['input'][:8]
    if next(model.parameters()).is_cuda:
        vis_input = vis_input.cuda()
    output, losses, stats, att_stats, comp_stats = model(vis_input)

    # Input and recon
    writer.add_image(mode+'_input', make_grid(vis_batch['input'][:8]), iter_idx)
    writer.add_image(mode+'_recon', make_grid(output), iter_idx)

    # Decomposition
    for key in ['mx_r_k', 'x_r_k', 'log_m_k', 'log_m_r_k']:
        if key not in stats:
            continue
        for step, val in enumerate(stats[key]):
            if 'log' in key:
                val = val.exp()
            writer.add_image(f'{mode}_{key}/k{step}', make_grid(val), iter_idx)


def evaluation(model, data_loader, writer, config, iter_idx, N_eval=None):
    # TODO(martin): make interface cleaner

    model.eval()

    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if iter_idx == 0 or config.debug:
        num_batches = 1
    elif N_eval is not None and N_eval <= len(data_loader)*data_loader.batch_size:
        num_batches = N_eval // data_loader.batch_size
        fprint(t + f" | Evaluating only on first {N_eval} examples in loader")
    else:
        num_batches = len(data_loader)
        fprint(t + f" | Evaluating on all {num_batches} examples in loader")

    start_t = time.time()
    err, kl_l, kl_m, elbo = 0., 0., 0., 0.
    batch = None

    # Don't compute gradient to run faster
    with torch.no_grad():
        # Loop over loader
        for b_idx, batch in enumerate(data_loader):
            if config.gpu:
                batch['input'] = batch['input'].cuda()
            if b_idx == num_batches:
                fprint(f"Breaking from eval loop after {b_idx} batches")
                break

            if b_idx % 100 == 0:
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fprint(t + f" | Validation batch [{b_idx+1} | {num_batches}]")

            _, losses, stats, _, _ = model(batch['input'])

            new_err = losses.err.mean(0)
            err += float(new_err) / num_batches
            # Parse different loss types
            if 'kl_m' in losses:
                new_kl_m = losses.kl_m.mean(0)
                kl_m += float(new_kl_m) / num_batches
            elif 'kl_m_k' in losses:
                new_kl_m = torch.stack(losses.kl_m_k, dim=1).sum(1).mean(0)
                kl_m += float(new_kl_m) / num_batches
            if 'kl_l' in losses:
                new_kl_l = losses.kl_l.mean(0)
                kl_l += float(new_kl_l) / num_batches
            elif 'kl_l_k' in losses:
                new_kl_l = torch.stack(losses.kl_l_k, dim=1).sum(1).mean(0)
                kl_l += float(new_kl_l) / num_batches
            # Update ELBO
            if 'elbo' not in losses:
                # Assign current "estimate"
                elbo += float(new_err + new_kl_l + new_kl_m) / num_batches
            else:
                # Add over steps
                elbo += float(losses.elbo.mean(0)) / num_batches

    # Printing
    duration = time.time() - start_t
    pstr = f'Evaluation elbo: {elbo:.1f}'
    pstr += f', err: {err:.1f}, kl_l: {kl_l:.1f}'
    pstr += f', kl_m: {kl_m:.1f}'
    pstr += f' --- {num_batches / duration:.1f} b/s'
    fprint(pstr)

    # TensorBoard logging
    if writer is not None:
        # TensorBoard logging - scalars
        writer.add_scalar('val/elbo', elbo, iter_idx)
        writer.add_scalar('val/err', err, iter_idx)
        writer.add_scalar('val/kl_l', kl_l, iter_idx)
        writer.add_scalar('val/kl_m', kl_m, iter_idx)
        # TensorBoard logging - inference (limit to 8)
        visualise_inference(model, batch, writer, 'val', iter_idx)
        # TensorBoard logging - generation (limit to 8)
        try:
            output, stats = model.sample(batch_size=8, K_steps=config.K_steps)
            writer.add_image('samples', make_grid(output), iter_idx)
            for key in ['x_k', 'log_m_k', 'mx_k']:
                if key not in stats:
                    continue
                for step, val in enumerate(stats[key]):
                    if 'log' in key:
                        val = val.exp()
                    writer.add_image(f'gen_{key}/k{step}', make_grid(val),
                                     iter_idx)
        except NotImplementedError:
            fprint("Sampling not implemented for this model.")

    model.train()

    return elbo


if __name__ == '__main__':
    main_flags()
    main()
