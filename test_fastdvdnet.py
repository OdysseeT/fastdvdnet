#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import batch_psnr, init_logger_test, \
variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger, get_imagenames

NUM_IN_FR_EXT = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

def save_out_seq(seqclean, save_dir, sigmaval, start_idx):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seqclean.size()[0]
    for idx in range(seq_len):
        save_img(seqclean, save_dir, sigmaval, start_idx,idx)

def save_img(seq, save_dir, sigmaval, start_idx, idx):
    out_name = os.path.join(save_dir,\
                ('n{}_DVDnet_{:0>15}').format(sigmaval, start_idx+idx) + OUTIMGEXT)

    outimg = variable_to_cv2_image(seq[idx].unsqueeze(dim=0))
    cv2.imwrite(out_name, outimg)

def test_fastdvdnet(**args):
    """Denoises all sequences present in a given folder. Sequences must be stored as numbered
    image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

    Inputs:
    args (dict) fields:
    "model_file": path to model
    "test_path": path to sequence to denoise
    "suffix": suffix to add to output name
    "max_num_fr_per_seq": max number of frames to load per sequence
    "noise_sigma": noise level used on test set
    "dont_save_results: if True, don't save output images
    "no_gpu": if True, run model on CPU
    "save_path": where to save outputs as png
    "gray": if True, perform denoising of grayscale images instead of RGB
    """
    # Start time
    start_time = time.time()

    # If save_path does not exist, create it
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

        # Sets data type according to CPU or GPU modes
    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

        # Create models
    print('Loading models ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(args['model_file'])
    if args['cuda']:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    nb_file = len(os.listdir(args['test_path']))
    # Get ordered list of filenames
    files = get_imagenames(args['test_path'])
    start_time = time.time()

    for fidx in range(0, len(files), args['max_num_fr_per_seq']):
        subfiles = files[fidx:fidx+args['max_num_fr_per_seq']]
        seq, _, _ = open_sequence(subfiles, args['test_path'],\
                        args['gray'],\
                        expand_if_needed=False,\
                        max_num_fr=args['max_num_fr_per_seq'])
        # process data
        with torch.no_grad():
            seq = torch.from_numpy(seq).to(device)
            seq_time = time.time()

            # Add noise
            noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
            seqn = seq + noise
            noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

            denframes = denoise_seq_fastdvdnet(seq=seqn,\
                                            noise_std=noisestd,\
                                            temp_psz=NUM_IN_FR_EXT,\
                                            model_temporal=model_temp)
        print("Saving files starting at index {}".format(fidx))
        save_out_seq(denframes, args['save_path'], int(args['noise_sigma']*255), fidx)
    print("denoise_seq_dvdnet done")
    den_time = time.time()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str,\
        default="./model.pth", \
        help='path to model of the pretrained denoiser')
    parser.add_argument("--test_path", type=str, default="./data/rgb/Kodak24", \
        help='path to sequence to denoise')
    parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=25, \
        help='max number of frames to load per sequence')
    parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--save_path", type=str, default='./results', \
        help='where to save outputs as png')
    parser.add_argument("--gray", action='store_true',\
        help='perform denoising of grayscale images instead of RGB')

    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))

    test_fastdvdnet(**vars(argspar))
