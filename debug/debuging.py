import os, sys
import torch
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from visualize import draw_dataloader_output
from data.dataloader import get_coco_dataloader, COCO_dataset
from data.transform import Resizer, ToTensor, collater, ToPIL, RetinaNet_transform


label_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
              'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
              'toothbrush']

def gt_debug(root,
             annFile):
    transform = transforms.Compose([Resizer(),
                                    ToTensor()])

    dataloader = get_coco_dataloader(data_dir=root,
                                     annotation_path=annFile,
                                     transform=transform,
                                     batch_size=4,
                                     num_workers=2,
                                     collater=collater)
    PIL_converter = ToPIL()
    for idx, data in enumerate(dataloader):
        images = data['image']
        images = PIL_converter(images)

        gt_loc = data['gt_loc'].numpy().tolist()
        gt_cls = data['gt_cls'].numpy().tolist()

        draw_dataloader_output(images,
                               gt_loc,
                               gt_cls)

def anchor_debug(root,
                 annFile):
    image_size = 512
    aspect_ratios = [1, 0.5, 2.0]
    scales = [2 ** i for i in [0, 1 / 3, 2 / 3]]
    strides = [2 ** i for i in range(3, 8)]
    areas = [32, 64, 128, 256, 512]
    transform = transforms.Compose([Resizer(),
                                    ToTensor()])

    dataloader = get_coco_dataloader(data_dir=root,
                                     annotation_path=annFile,
                                     transform=transform,
                                     batch_size=128,
                                     num_workers=2,
                                     collater=collater)

    PIL_converter = ToPIL()
    retinanet_transform = RetinaNet_transform(image_height=image_size,
                                              image_width=image_size,
                                              num_classes=len(label_list),
                                              aspect_ratios=aspect_ratios,
                                              scales=scales,
                                              strides=strides,
                                              areas=areas,
                                              iou_threshold=0.5)
    for idx, data in enumerate(dataloader):
        images = data['image']
        images = PIL_converter(images)

        gt_loc = data['gt_loc']
        gt_cls = data['gt_cls']

        for i in range(gt_loc.shape[0]):
            encode_cls, encode_loc = retinanet_transform.encode(gt_cls[i], gt_loc[i])
            scores, categories, pred_loc = retinanet_transform.decode(encode_cls, encode_loc)

            anchor_loc = pred_loc[scores != 0.5]
            anchor_cls = categories[scores != 0.5]
            print(torch.top_k(scores, 3))
            print(categories.shape)
            exit()
            # print(anchor_loc)
            # print(anchor_cls)
        exit()


if __name__ == "__main__":
    root = '/home/jm/research/COCO/val2017'
    annFile = '/home/jm/research/COCO/annotations/instances_val2017.json'

    anchor_debug(root, annFile)

