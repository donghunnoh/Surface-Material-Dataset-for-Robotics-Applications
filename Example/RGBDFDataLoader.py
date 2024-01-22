import os

import pdb

import copy

import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from pycocotools.coco import COCO
from pycocotools import mask

from tqdm import trange


import matplotlib.pyplot as plt
import skimage.io as io

class RGBDFDatasetSegmentation(Dataset):
    """RGBDF dataset in COCO format. Compatible with torch.utils.data.DataLoader
    """
    def __init__(self, root, json, transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), class_layers=False):
        self.root = root
        self.coco = COCO(root+json)
        self.class_layers = class_layers

        # ----- Categories
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = self.coco.getCatIds()
        self.cat_ids_idx = {}
        for idx in range(len(self.cat_ids)):
            self.cat_ids_idx[self.cat_ids[idx]] = idx

        # ----- File paths
        ids_file = os.path.join(self.root, 'annotations/train_ids.pth')

        # ----- Load pre-processed IDs file or create the IDS file
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            img_keys = list(self.coco.imgs.keys())
            self.ids = self._preprocess(img_keys, ids_file)
        
        # ----- Transforms
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        """
        Returns:
            input_img:
            target_img:
            input_ir:
            target_ir:
            input_depth:
            target_depth:
        """
        coco = self.coco # Relabel for convenience

        # ----- 1. Get the image the index is referring to
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        fn_type = img_metadata['file_name']
        fn_rgb = 'image_RGB_'+fn_type[-12:]
        fn_ir = 'image_IR_'+fn_type[-12:]
        fn_depth = 'image_DEPTH_'+fn_type[-12:]

        # ----- 1.2 Find paths to RGB, IR, DEPTH
        # TODO: Find another way to get only ImgIds of the exact catIds
        img_id_all_inclusive = self.coco.getImgIds(catIds=img_metadata['category_ids'])
        img_metadata_all_inclusive = coco.loadImgs(img_id_all_inclusive)

        img_id_all = []
        img_metadata_all = []
        if len(img_id_all_inclusive) > 300:
            for idx in range(len(img_id_all_inclusive)):
                if img_metadata_all_inclusive[idx]['category_ids'] == img_metadata['category_ids']:
                    img_id_all.append(img_metadata_all_inclusive[idx]['id'])
                    img_metadata_all.append(img_metadata_all_inclusive[idx])
        else:
            img_id_all = img_id_all_inclusive
            img_metadata_all = img_metadata_all_inclusive

        assert(len(img_id_all)==300),"# of images per material should be 300!"
        
        img_metadata_rgb, img_metadata_ir, img_metadata_depth = None, None, None
        for img_select in img_metadata_all:
            if img_select['file_name'] == fn_rgb:
                img_metadata_rgb = img_select
            if img_select['file_name'] == fn_ir:
                img_metadata_ir = img_select
            if img_select['file_name'] == fn_depth:
                img_metadata_depth = img_select
            if img_metadata_rgb != None and img_metadata_ir != None and img_metadata_depth != None:
                break

        assert(img_metadata_rgb != None),"No RGB found!"
        assert(img_metadata_ir != None),"No IR found!"
        assert(img_metadata_depth != None),"No DEPTH found!"

        # ----- 2. Get all imgs and ground truth masks
        img_rgb, target_rgb, mask_np_rgb = self._get_img_gt(img_metadata_rgb)
        img_ir, target_ir, mask_np_ir = self._get_img_gt(img_metadata_ir)
        img_depth, target_depth, raw_mask = self._get_img_gt(img_metadata_depth)

        # ----- 3. Generate friction input and mask
        cat_id_mask = np.unique(raw_mask).tolist()
        cat_id_meta = [0]+img_metadata['category_ids']
        cat_id_mask.sort()
        cat_id_meta.sort()
        assert(cat_id_mask==cat_id_meta),"Raw mask does not have all the categories!"

        # ----- 3.1 Get the friction coefficients
        ann_metadata = coco.loadAnns(coco.getAnnIds(imgIds=img_metadata_depth['id']))
        if self.class_layers:
            target_fric = copy.deepcopy(raw_mask).astype(float)
            for idx in range(len(ann_metadata)):
                ann_cat_id = ann_metadata[idx]['category_id']
                target_fric[target_fric==ann_cat_id] = ann_metadata[idx]['friction_coeff'][str(ann_cat_id)]
            img_fric = np.zeros(img_depth.size[::-1])
            for catx in cat_id_mask[1:]:
                unique_vals = np.unique(target_fric[self.cat_ids_idx[catx],:,:])
                assert(len(unique_vals)==2),"There should only be two numbers in a layer with labels!"
                img_fric[target_fric[self.cat_ids_idx[catx],:,:]!=0] = unique_vals[1]
            img_fric = Image.fromarray(img_fric)
            target_fric = torch.from_numpy(target_fric)
        else:
            img_fric = copy.deepcopy(raw_mask).astype(float)
            for idx in range(len(ann_metadata)):
                ann_cat_id = ann_metadata[idx]['category_id']
                img_fric[img_fric==ann_cat_id] = ann_metadata[idx]['friction_coeff'][str(ann_cat_id)]
            img_fric = Image.fromarray(img_fric)
            target_fric = copy.deepcopy(img_fric)

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_ir = self.transform(img_ir)
            img_depth = self.transform(img_depth)
            img_fric = self.transform(img_fric)

        # TODO: Support cropping for tensors
        # -> by defining additional transform transforms.ToPILImage()?
        if self.target_transform and not self.class_layers:
            target_rgb = self.target_transform(target_rgb)
            target_ir = self.target_transform(target_ir)
            target_depth = self.target_transform(target_depth)
            target_fric = self.target_transform(target_fric)

        return img_rgb, target_rgb, img_ir, target_ir, img_depth, target_depth, img_fric, target_fric
    
    def __len__(self):
        return len(self.ids)
    
    def _get_img_gt(self, img_metadata):
        # ----- 1. If using data from local server
        img = Image.open('.'+img_metadata['path'])
        # -------- 1.1 If using RoMeLa server
        # response = requests.get(path)
        # img = Image.open(BytesIO(response.content)).convert('RGB')
        # img = Image.open(os.path.join())self.ids

        # ----- 2. Get ground truth
        ann_ids = self.coco.getAnnIds(imgIds=img_metadata['id'])
        target = self.coco.loadAnns(ann_ids)

        # ----- 4. Get masks
        if self.class_layers:
            target_np = self._generate_masks_layered(
                target, img_metadata['height'], img_metadata['width']
            )
            target = torch.from_numpy(target_np)
        else:
            target_np = self._generate_masks(
                target, img_metadata['height'], img_metadata['width']
            )
            target = Image.fromarray(target_np)            

        return img, target, target_np
    
    def _generate_masks(self, target, h, w):
        maskedTarget = np.zeros((h, w), dtype=np.uint8)

        for instance in target:
            rle = mask.frPyObjects(instance['segmentation'], h, w)
            m = mask.decode(rle)
            cat_id = instance['category_id']
            if cat_id in self.cat_ids:
                # c = self.cat_ids.index(cat_id) # Original
                c = cat_id
            else:
                continue
            if len(m.shape) < 3:
                maskedTarget[:, :] += (maskedTarget == 0) * (m * c)
            else:
                maskedTarget[:, :] += (maskedTarget == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return maskedTarget
    
    def _generate_masks_layered(self, target, h, w):
        maskedTarget = np.zeros((len(self.cat_ids), h, w), dtype=np.uint8)

        for instance in target:
            rle = mask.frPyObjects(instance['segmentation'], h, w)
            m = mask.decode(rle)
            cat_id = instance['category_id']
            if cat_id in self.cat_ids:
                # c = self.cat_ids.index(cat_id) # Original
                c = cat_id
            else:
                continue
            maskedTarget[self.cat_ids_idx[cat_id], :, :] = (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)

        return maskedTarget
    
    def _preprocess(self, ids, ids_file):
        print("Pre-processing masks!")

        tbar = trange(len(ids))
        new_ids = []
        for idx in tbar:
            img_id = ids[idx]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)) # Get annotations with image_id == img_id
            img_metadata = self.coco.loadImgs(img_id)[0]
            target_mask = self._generate_masks(cocotarget, img_metadata['height'], img_metadata['width'])

            if (target_mask > 0).sum() > 1000:
                new_ids.append(img_id)
            else:
                print("Not over 1000!")
            tbar.set_description("Doing: {} / {}, got {} qualified images.".format(idx, len(ids), len(new_ids)))

        print("Found number of qualified images: {}".format(len(new_ids)))

        torch.save(new_ids, ids_file)

        return new_ids
    
    def _replace_mask_friction(self, maskarr):
        pass


def get_loader(root, json, transform, target_transform, batch_size, shuffle, num_workers, class_layers, val_amount=0.0):
    assert(1.0-val_amount>0),"Resize your validation set and test set."
    rgbdf = RGBDFDatasetSegmentation(root=root, json=json, transform=transform, target_transform=target_transform, class_layers=class_layers)
    n_rgbdf = len(rgbdf)
    n_val = int(n_rgbdf*val_amount)
    n_train = n_rgbdf - n_val
    train_set, val_set = torch.utils.data.random_split(rgbdf, (n_train, n_val))

    # Data loader for COCO dataset
    if val_amount > 0:
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)
        return train_loader, val_loader
    else:
        data_loader = torch.utils.data.DataLoader(dataset=rgbdf,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
        return data_loader


if __name__ == '__main__':
    RD = RGBDFDatasetSegmentation('datasets/', 'RGBDF.json', class_layers=False)
    RDtest = RGBDFDatasetSegmentation('datasets/', 'RGBDF_testset.json', class_layers=False)

    complete_loader_RGB = get_loader(root='datasets/', 
                                     json='RGBDF.json', 
                                     transform=transforms.ToTensor(), 
                                     target_transform=transforms.ToTensor(), 
                                     batch_size=1, 
                                     shuffle=True, 
                                     num_workers=1,
                                     class_layers=False,
                                     val_amount=0.0)
    
    trainSet, valSet = get_loader(root='datasets/', 
                                  json='RGBDF.json', 
                                  transform=transforms.ToTensor(), 
                                  target_transform=transforms.ToTensor(), 
                                  batch_size=1, 
                                  shuffle=True, 
                                  num_workers=1,
                                  class_layers=False,
                                  val_amount=0.1)
    
    complete_loader_RGB = get_loader(root='datasets/', 
                                     json='RGBDF_testset.json', 
                                     transform=transforms.ToTensor(), 
                                     target_transform=transforms.ToTensor(), 
                                     batch_size=1, 
                                     shuffle=True, 
                                     num_workers=1,
                                     class_layers=False,
                                     val_amount=0.0)

    for idx in range(len(RD.ids)):
        print("idx: {}".format(idx))
        RD.__getitem__(idx)