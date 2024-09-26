#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import pickle
import numpy as np
import torchvision

from engines import CoreEngine

class Tracker:
    def __init__(self, focal_length, device='cuda'):
        self._device = device
        self.tracker = CoreEngine(focal_length=focal_length, device=device)

    def track_image(self, image_path, no_crop=False, no_matting=False):
        # build name
        output_path = 'outputs/{}'.format(os.path.basename(image_path))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_tensor = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        image_tensor = image_tensor.to(self._device).float()
        print('Track image...')
        if_crop, if_matting = not no_crop, not no_matting
        track_results, vis_results = self.tracker.track_image(
            image_tensor, if_crop=if_crop, if_matting=if_matting
        )
        if track_results is not None:
            with open(os.path.join(output_path, 'optim.pkl'), 'wb') as f:
                pickle.dump(track_results, f)
            torchvision.utils.save_image([
                    torch.tensor(track_results['image']), vis_results
                ], os.path.join(output_path, 'track.jpg')
            )
            print('Track done: {}!'.format(output_path))
        else:
            print('No face detected!')


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--no_matting', action='store_true')
    parser.add_argument('--no_crop', action='store_true')
    args = parser.parse_args()
    
    tracker = Tracker(focal_length=12.0, device='cuda')
    tracker.track_image(args.image_path, no_crop=args.no_crop, no_matting=args.no_matting)
