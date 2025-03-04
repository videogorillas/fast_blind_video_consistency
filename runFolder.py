#!/usr/bin/python

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
# import numpy as np

### torch lib
import torch
# import torch.nn as nn

### custom lib
# from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    ### dataset options
    parser.add_argument('--origin', type=str, default='/svlk/bab5_s01e15_first25/', help='origin')
    parser.add_argument('--input', type=str, default="/svlk/bab5_s01e15_first25_SR/", help='blinked source')
    parser.add_argument('--output', type=str, default='/svlk/bab5_s01e15_first25_SR_stable/', help='output')
    parser.add_argument('--first', type=int, default=1, help='firtst frame number')
    parser.add_argument('--last', type=int, default=1, help='last frame number')
    ### other options
    parser.add_argument('-gpu', type=int, default=0, help='gpu device id')

    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 2  ## Inputs to TransformNet need to be divided by 4

    ### load model opts
    opts_filename = os.path.join('pretrained_models', "ECCV18_blind_consistency_opts.pth")
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)

    ### initialize model
    print('===> Initializing model from %s...' % model_opts.model)
    model = networks.__dict__[model_opts.model](model_opts, nc_in=12, nc_out=3)

    ### load trained model
    model_filename = os.path.join('pretrained_models', "ECCV18_blind_consistency.pth")
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict['model'])

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)

    model.eval()

    times = []

    if not os.path.isdir(opts.output):
        os.makedirs(opts.output)

    # full_list = glob.glob(sorted(os.path.join(opts.origin, "*.png")))
    # output_list = glob.glob(os.path.join(opts.output, "*.png"))
    frames_list = []
    for f in range(opts.first, opts.last):
        frames_list.append(f)

    ## frame 0
    frame_p1 = utils.read_img(os.path.join(opts.input, '{:06d}.png'.format(frames_list[0])))
    output_filename = os.path.join(opts.output, '{:06d}.png'.format(frames_list[0]))
    utils.save_img(frame_p1, output_filename)

    lstm_state = None

    for t in range(1, len(frames_list)):
        ### load frames
        frame_origin_1 = utils.read_img(os.path.join(opts.origin, '{:06d}.png'.format(frames_list[t-1])))
        frame_origin_2 = utils.read_img(os.path.join(opts.origin, '{:06d}.png'.format(frames_list[t])))
        frame_output_1 = utils.read_img(os.path.join(opts.output, '{:06d}.png'.format(frames_list[t-1])))
        frame_input_2 = utils.read_img(os.path.join(opts.input, '{:06d}.png'.format(frames_list[t])))

        ### resize image
        H_orig = frame_input_2.shape[0]
        W_orig = frame_input_2.shape[1]

        H_sc = int(math.ceil(float(H_orig) / opts.size_multiplier) * opts.size_multiplier)
        W_sc = int(math.ceil(float(W_orig) / opts.size_multiplier) * opts.size_multiplier)

        frame_origin_1 = cv2.resize(frame_origin_1, (W_sc, H_sc))
        frame_origin_2 = cv2.resize(frame_origin_2, (W_sc, H_sc))
        frame_output_1 = cv2.resize(frame_output_1, (W_sc, H_sc))
        frame_input_2 = cv2.resize(frame_input_2, (W_sc, H_sc))

        with torch.no_grad():
            ### convert to tensor
            frame_origin_1 = utils.img2tensor(frame_origin_1).to(device)
            frame_origin_2 = utils.img2tensor(frame_origin_2).to(device)
            frame_output_1 = utils.img2tensor(frame_output_1).to(device)
            frame_input_2 = utils.img2tensor(frame_input_2).to(device)

            ### model input
            inputs = torch.cat((frame_input_2, frame_output_1, frame_origin_2, frame_origin_1), dim=1)

            ### forward
            ts = time.time()

            output, lstm_state = model(inputs, lstm_state)
            frame_o2 = frame_input_2 + output

            te = time.time()
            times.append(te - ts)

            ## create new variable to detach from graph and avoid memory accumulation
            lstm_state = utils.repackage_hidden(lstm_state)

        ### convert to numpy array
        frame_o2 = utils.tensor2img(frame_o2)

        ### resize to original size
        frame_o2 = cv2.resize(frame_o2, (W_orig, H_orig))

        ### save output frame
        output_filename = os.path.join(opts.output, '{:06d}.png'.format(frames_list[t]))
        utils.save_img(frame_o2, output_filename)

    ## end of frame
    ## end of video
