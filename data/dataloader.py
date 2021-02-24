import os
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np
import random

from bbox.box_utils import coco_to_conner_form


class COCO_dataset(Dataset):
    def __init__(self,
                 root: str,
                 annFile: str,
                 transform: Optional[Callable] = None,
                 validation = False) -> None:
        super(COCO_dataset, self).__init__()
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.validation = validation
        random.shuffle(self.ids)

    def __getitem__(self, index: int) ->Tuple[Any, Any]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        gt_loc, gt_cls = self._get_gt(target)
        if self.transform is not None:
            data={'image':img,
                  'gt_loc': gt_loc,
                  'gt_cls': gt_cls}
            if self.validation: # only for coco evaluation
                data['img_id'] = img_id
                # data['img_path'] = os.path.join(self.root, path)
                width, height = img.size
                data['img_size'] = np.array([width, height])
            data = self.transform(data)
            return data
        return img, target

    def _get_gt(self, target):
        gt_loc = []
        gt_cls = []
        for i in range(len(target)):
            gt_loc.append(coco_to_conner_form(target[i]['bbox']))
            gt_cls.append(target[i]['category_id'] - 1)
        return gt_loc, gt_cls


    def __len__(self) -> int:
        return len(self.ids)


def get_coco_dataloader(data_dir,
                        annotation_path,
                        transform,
                        collater,
                        batch_size=1,
                        num_workers=1,
                        validation=False):
    dataset = COCO_dataset(root=data_dir,
                           annFile=annotation_path,
                           transform=transform,
                           validation=validation)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=collater)

    return dataloader