#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
import os
import sys
import torch
import pickle
import random
import shutil
import numpy as np
import torchvision
from tqdm.rich import tqdm

from .utils_lmdb import LMDBEngine
from .engine_optim import OptimEngine
from .vgghead_detector import VGGHeadDetector
from .flame_model import FLAMEModel, RenderMesh
from .emica_encoder import ImageEngine, EmicaEncoder
from .human_matting import StyleMatteEngine as HumanMattingEngine

class CoreEngine:
    def __init__(self, focal_length, device='cuda'):
        random.seed(42)
        self._device = device
        # paths and data engine
        self.emica_encoder = EmicaEncoder(device=device)
        self.emica_data_engine = ImageEngine(device=device)
        self.vgghead_encoder = VGGHeadDetector(device=device)
        self.matting_engine = HumanMattingEngine(device=device)
        calibration_results = {'focal_length':focal_length, 'verts_scale': 5.0}
        self.calibration_results = calibration_results
        self.optim_engine = OptimEngine(self.calibration_results, device=device)

    def build_video(self, video_path, output_path, matting=False, background=0.0):
        video_name = os.path.basename(video_path).split('.')[0]
        if os.path.exists(output_path):
            print(f'Output path {output_path} exists, replace it.')
            shutil.rmtree(output_path) 
        os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'img_lmdb')):
            lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=True)
            frames_data, _, meta_data = torchvision.io.read_video(video_path, output_format='TCHW')
            assert frames_data.shape[0] > 0, 'No frames in the video, reading video failed.'
            print(f'Processing video {video_path} with {frames_data.shape[0]} frames.')
            for fidx, frame in tqdm(enumerate(frames_data), total=frames_data.shape[0], ncols=80, colour='#95bb72'):
                if meta_data['video_fps'] > 50:
                    if fidx % 2 == 0:
                        continue
                frame = torchvision.transforms.functional.resize(frame, 512, antialias=True) 
                frame = torchvision.transforms.functional.center_crop(frame, 512)
                if matting:
                    frame = self.matting_engine(
                        frame/255.0, return_type='matting', background_rgb=background
                    ).cpu()*255.0
                lmdb_engine.dump(f'{video_name}_{fidx}', payload=frame, type='image')
            lmdb_engine.random_visualize(os.path.join(output_path, 'img_lmdb', 'visualize.jpg'))
            lmdb_engine.close()
            return meta_data['video_fps']
        else:
            video_reader = torchvision.io.VideoReader(src=video_path)
            meta_data = video_reader.get_metadata()['video']
            return meta_data['fps'][0]

    def track_base(self, lmdb_engine, output_path):
        if os.path.exists(os.path.join(output_path, 'base.pkl')):
            with open(os.path.join(output_path, 'base.pkl'), 'rb') as f:
                base_results = pickle.load(f)
            return base_results
        else:
            images_dataset = ImagesData(lmdb_engine)
            images_loader = torch.utils.data.DataLoader(
                images_dataset, batch_size=1, num_workers=2, shuffle=False
            )
            images_loader = iter(images_loader)
            base_results = {}
            for image_data in tqdm(images_loader, ncols=80, colour='#95bb72'):
                image_data = data_to_device(image_data, device=self._device)
                image, image_key = image_data['image'][0], image_data['image_key'][0]
                emica_inputs = self.emica_data_engine(image, image_key)
                if emica_inputs is None:
                    continue
                emica_inputs = torch.utils.data.default_collate([emica_inputs])
                emica_inputs = data_to_device(emica_inputs, device=self._device)
                emica_results = self.emica_encoder(emica_inputs)
                vgg_results, bbox, lmks_2d70 = self.vgghead_encoder(image, image_key)
                if vgg_results is None:
                    continue
                emica_results, vgg_results = self._process_emica_vgg(emica_results, vgg_results, lmks_2d70)
                base_results[image_key] = {
                    'emica_results': emica_results, 
                    'vgg_results': vgg_results, 
                    'bbox': bbox.cpu().numpy() / 512.0
                }
            with open(os.path.join(output_path, 'base.pkl'), 'wb') as f:
                pickle.dump(base_results, f)
            return base_results

    def track_optim(self, base_result, output_path, lmdb_engine=None, share_id=False):
        if os.path.exists(os.path.join(output_path, 'optim.pkl')):
            with open(os.path.join(output_path, 'optim.pkl'), 'rb') as f:
                optim_results = pickle.load(f)
            return optim_results
        else:
            # self.optim_engine.init_model(self.calibration_results, image_size=512)
            base_result = {k: v for k, v in base_result.items() if v is not None}
            mini_batchs = build_minibatch(list(base_result.keys()), share_id=share_id)
            if lmdb_engine is not None:
                batch_frames = torch.stack([lmdb_engine[key] for key in mini_batchs[0][:20]]).to(self._device).float()
            else:
                batch_frames = None
            optim_results = {}
            for mini_batch in tqdm(mini_batchs, ncols=80, colour='#95bb72'):
                mini_batch_emica = [base_result[key] for key in mini_batch]
                mini_batch_emica = torch.utils.data.default_collate(mini_batch_emica)
                mini_batch_emica = data_to_device(mini_batch_emica, device=self._device)
                optim_result, visualization = self.optim_engine.lightning_optimize(
                    mini_batch, mini_batch_emica, batch_frames=batch_frames, share_id=share_id
                )
                batch_frames = None
                if visualization is not None:
                    torchvision.utils.save_image(visualization, os.path.join(output_path, 'optim.jpg'))
                optim_results.update(optim_result)
            with open(os.path.join(output_path, 'optim.pkl'), 'wb') as f:
                pickle.dump(optim_results, f)
            return optim_results

    def track_image(self, inp_image, if_crop=True, if_matting=True):
        assert inp_image.dim() == 3, f'Image dim must be 3, but got {inp_image.dim()}.'
        assert inp_image.max() > 1.0, f'Image in [0, 255.0], but got {inp_image.max()}.'
        if if_crop:
            inp_image = self.crop_image(inp_image)
            if inp_image is None:
                return None, None
        if if_matting:
            inp_image = self.matting_engine.forward(inp_image/255.0, return_type='matting', background_rgb=0.0).clamp(0.0, 1.0)
            inp_image = inp_image * 255.0
        emica_inputs = self.emica_data_engine(inp_image, 'online_track')
        if emica_inputs is None:
            return None, None
        emica_inputs = torch.utils.data.default_collate([emica_inputs])
        emica_inputs = data_to_device(emica_inputs, device=self._device)
        emica_results = self.emica_encoder(emica_inputs)
        vgg_results, bbox, lmks_2d70 = self.vgghead_encoder(inp_image, 'online_track')
        if vgg_results is None:
            return None, None
        emica_results, vgg_results = self._process_emica_vgg(emica_results, vgg_results, lmks_2d70)
        base_results = {
            'emica_results': emica_results, 'vgg_results': vgg_results,
            'bbox': bbox.cpu().numpy() / 512.0
        }
        base_results = torch.utils.data.default_collate([base_results])
        base_results = data_to_device(base_results, device=self._device)
        track_results, vis_results = self.optim_engine.lightning_optimize(
            ['online_track'], base_results, batch_frames=inp_image[None]
        )
        track_results['online_track']['image'] = (inp_image / 255.0).cpu().numpy()
        return track_results['online_track'], vis_results

    def crop_image(self, inp_image):
        ori_height, ori_width = inp_image.shape[1:]
        if not hasattr(self.emica_data_engine, 'insight_detector'):
            self.emica_data_engine._init_models()
        bbox, _, _ = self.emica_data_engine.insight_detector.get(inp_image)
        # _, bbox, _ = self.vgghead_encoder(inp_image, 'online_track')
        if bbox is None:
            return None
        bbox = expand_bbox(bbox, scale=1.85).long()
        crop_image = torchvision.transforms.functional.crop(
            inp_image, top=bbox[1], left=bbox[0], height=bbox[3]-bbox[1], width=bbox[2]-bbox[0]
        )
        crop_image = torchvision.transforms.functional.resize(crop_image, (512, 512), antialias=True)
        return crop_image

    @staticmethod
    def _process_emica_vgg(emica_results, vgg_results, lmks_2d70):
        processed_emica_results = {
            'shapecode': emica_results['shapecode'][0].cpu().numpy(),
            'expcode': emica_results['expcode'][0].cpu().numpy(),
            'globalpose': emica_results['globalpose'][0].cpu().numpy(),
            'jawpose': emica_results['jawpose'][0].cpu().numpy(),
        }
        processed_vgg_results = {
            'shapecode': vgg_results['shapecode'].cpu().numpy(),
            'expcode': vgg_results['expcode'].cpu().numpy(),
            'posecode': vgg_results['posecode'].cpu().numpy(),
            'transform': {
                'rotation_6d': vgg_results['rotation_6d'].cpu().numpy(),
                'translation': vgg_results['translation'].cpu().numpy(),
                'scale': vgg_results['scale'].cpu().numpy(),
            },
            'normalize': vgg_results['normalize'],
            'lmks_2d70': lmks_2d70.cpu().numpy(),
        }
        return processed_emica_results, processed_vgg_results


class ImagesData(torch.utils.data.Dataset):
    def __init__(self, lmdb_engine):
        super().__init__()
        self._lmdb_engine = lmdb_engine
        self._image_keys = list(lmdb_engine.keys())

    def __getitem__(self, index):
        image_key = self._image_keys[index]
        image = self._lmdb_engine[image_key]
        return {'image': image, 'image_key': image_key}

    def __len__(self, ):
        return len(self._image_keys)


def data_to_device(data_dict, device='cuda'):
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.tensor(data_dict[key], device=device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_device(data_dict[key], device=device)
        else:
            continue
    return data_dict


def build_minibatch(all_frames, batch_size=1024, share_id=False):
    if share_id:
        all_frames = sorted(all_frames)
        video_names = list(set(['_'.join(frame_name.split('_')[:-1]) for frame_name in all_frames]))
        video_frames = {video_name: [] for video_name in video_names}
        for frame in all_frames:
            video_name = '_'.join(frame.split('_')[:-1])
            video_frames[video_name].append(frame)
        all_mini_batch = []
        for video_name in video_names:
            mini_batch = []
            for frame_name in video_frames[video_name]:
                mini_batch.append(frame_name)
                if len(mini_batch) % batch_size == 0:
                    all_mini_batch.append(mini_batch)
                    mini_batch = []
            if len(mini_batch):
                all_mini_batch.append(mini_batch)
    else:
        all_frames = sorted(all_frames, key=lambda x: int(x.split('_')[-1]))
        all_mini_batch, mini_batch = [], []
        for frame_name in all_frames:
            mini_batch.append(frame_name)
            if len(mini_batch) % batch_size == 0:
                all_mini_batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch):
            all_mini_batch.append(mini_batch)
    return all_mini_batch


def expand_bbox(bbox, scale=1.1):
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    cenx, ceny = (xmin + xmax) / 2, (ymin + ymax) / 2
    ceny = ceny - (ymax - ymin) * 0.05
    extend_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    xmine, xmaxe = cenx - extend_size / 2, cenx + extend_size / 2
    ymine, ymaxe = ceny - extend_size / 2, ceny + extend_size / 2
    expanded_bbox = torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
    return torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
