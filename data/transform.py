import torch
import numpy as np
from PIL import Image

from bbox.anchor import Anchor
from bbox.box_utils import compute_iou, rescale_bbox

class RetinaNet_transform():
    def __init__(self,
                 image_height,
                 image_width,
                 num_classes,
                 aspect_ratios,
                 scales,
                 strides,
                 areas,
                 iou_threshold):
        anchor_gen = Anchor(aspect_ratios, scales, strides, areas)
        self.anchors = anchor_gen.get_anchors(image_width, image_height)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold


    def encode(self, gt_cls, gt_loc):
        ## 1. compute iou and convert gt_loc to anchor form([num_anchors, 4])
        iou = compute_iou(self.anchors, gt_loc)  # [num_anchors, num_object]
        iou, max_idx = iou.max(dim=-1)
        gt_loc = gt_loc[max_idx]  # [num_anchors, 4]

        ## 2. convert gt_cls to anchor form([num_anchors, num_classes])
        conf = gt_cls[max_idx] # [num_anchors, ]
        conf[iou < self.iou_threshold] = self.num_classes # background
        # class index to one-hot vectors
        one_hot = torch.eye(self.num_classes + 1)
        one_hot = one_hot[conf.long()]  # [num_anchors, num_classes]
        one_hot = one_hot[:, :-1] # delete background

        ## 3. normalize gt_loc
        gt_loc[..., :2] = (gt_loc[..., :2] - self.anchors[..., :2]) / self.anchors[..., 2:]
        gt_loc[..., 2:] = torch.log((gt_loc[..., 2:] / self.anchors[..., 2:]) + 0.00001)

        return one_hot, gt_loc

    def decode(self, pred_cls, pred_loc):
        ## 1. unnormalize pred_loc
        pred_loc[..., :2] = (pred_loc[..., :2] * self.anchors[..., 2:]) + self.anchors[..., :2]
        pred_loc[..., 2:] = torch.exp(pred_loc[..., 2:]) * self.anchors[..., 2:]

        ## 2. convert pred_cls
        # pred_cls = torch.nn.functional.softmax(pred_cls, dim=-1)
        pred_cls = torch.sigmoid(pred_cls)
        scores, categories = pred_cls.max(dim=-1)

        return scores, categories, pred_loc


class Resizer():
    def __init__(self,
                 image_width=512,
                 image_height=512):
        self.image_height = image_height
        self.image_width = image_width
    def __call__(self, data):
        image = data['image']
        gt_loc = np.array(data['gt_loc']).astype(np.float32)
        width, height = image.size
        image = image.resize((self.image_width, self.image_height))
        gt_loc = rescale_bbox(gt_loc,
                              model_image_size=(self.image_width, self.image_height),
                              origin_image_size=(width, height))

        data['image'] = image
        data['gt_loc'] = gt_loc

        return data

class ToTensor():
    def __call__(self, data):
        image = np.array(data['image']).astype(np.float32)
        image = image / 255.
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        gt_loc = data['gt_loc']
        gt_loc = torch.from_numpy(gt_loc)
        gt_cls = np.array(data['gt_cls']).astype(np.float32)
        gt_cls = torch.from_numpy(gt_cls)

        data['image'] = image
        data['gt_loc'] = gt_loc
        data['gt_cls'] = gt_cls

        return data

class ToPIL():
    def __call__(self, images):
        images = images.numpy() * 255.
        images = images.transpose((0, 2, 3, 1))
        pil_images = []
        for img in images:
            img = Image.fromarray(img.astype(np.uint8))
            pil_images.append(img)

        return pil_images


def collater(data):
    images = [s['image'] for s in data]
    gt_loc = [s['gt_loc'] for s in data]
    gt_cls = [s['gt_cls'] for s in data]

    images = torch.stack(images, axis=0)

    max_num_annots = max(bbox.shape[0] for bbox in gt_loc)
    if max_num_annots > 0:
        bbox_padded = torch.ones((len(gt_loc), max_num_annots, 4)) * -1
        category_padded = torch.ones((len(gt_cls), max_num_annots)) * -1

        for i in range(len(gt_loc)):
            bbox = gt_loc[i]
            category = gt_cls[i]
            if bbox.shape[0] > 0:
                bbox_padded[i, :bbox.shape[0], :] = bbox
                category_padded[i, :category.shape[0]] = category

    else:
        bbox_padded = torch.ones((len(gt_loc), 1, 4)) * -1
        category_padded = torch.ones((len(gt_cls), 1, 1)) * -1

    return {'image': images, 'gt_loc': bbox_padded, 'gt_cls': category_padded}

