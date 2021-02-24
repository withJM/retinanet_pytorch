import torch

image_size = 512
aspect_ratios = [1, 0.5, 2.0]
scales = [2**i for i in [0, 1/3, 2/3]]
stride = [2**i for i in range(3, 8)]
areas = [32, 64, 128, 256, 512]
## featuremap_sizes = [64, 32, 16, 8, 4]

class Anchor():
    def __init__(self,
                 aspect_ratios,
                 scales,
                 strides,
                 areas):
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.strides = strides
        self.areas = areas

    def get_anchors(self,
                    image_width,
                    image_height):
        anchor_list = []
        for i in range(3, 8): # 8 ~ 128
            featuremap_width = image_width // (2**i) #64 ~ 4
            featuremap_height = image_height // (2**i) #64 ~ 4
            anchor_list.append(self._make_anchor(featuremap_width,
                                                 featuremap_height,
                                                 fpn_level=i).reshape([-1, 4])) # fpn_level = 3~7
        anchor = torch.cat(anchor_list, dim=0)
        return anchor

    def _make_anchor(self,
                     featuremap_width,
                     featuremap_height,
                     fpn_level):

        ### 1) make yx grid : [y, x, 2] = [h, w, 2]
        ## yx grid : [h, w, 2] ##
        num_anchors = len(self.scales) + len(self.aspect_ratios)
        grid_x = torch.arange(0.5, featuremap_width)
        grid_y = torch.arange(0.5, featuremap_height)

        grid_yx = torch.meshgrid(grid_x, grid_y)
        grid_yx = torch.stack(grid_yx, dim=-1)
        grid_yx = grid_yx.permute(1, 0, 2) # [x, y, 4] => [y, x, 4]
        # grid_yx = grid_yx.reshape(-1, 4)

        stride = self.strides[fpn_level-3]
        grid_yx = grid_yx.unsqueeze(0)
        grid_yx = torch.cat(num_anchors * [grid_yx]) * stride


        ### 2) make wh grid : [y, x, 2] = [h, w, 2]
        ## hw grid : [h, w, 2] ##
        num_anchors = len(self.aspect_ratios)
        area = self.areas[fpn_level-3]
        grid_wh_shape = [num_anchors, featuremap_height, featuremap_width, 2]
        grid_wh = torch.empty(grid_wh_shape).fill_(area)  # shape=[num_anchors, featuremap_dim, featuremap_dim, 2]
        aspect_ratio = torch.Tensor(self.aspect_ratios).unsqueeze(-1).unsqueeze(-1)
        grid_wh[..., 0] = grid_wh[..., 0] * torch.sqrt(aspect_ratio)  # w = scale * sqrt(ratio)
        grid_wh[..., 1] = grid_wh[..., 1] / torch.sqrt(aspect_ratio)  # h = scale / sqrt(ratio)


        ### 3) make scale wh grid : [y, x, 2] = [h, w, 2]
        ## for scale grid : [h, w, 2] ##
        num_anchors = len(self.scales)
        grid_scale_wh_shape = [num_anchors, featuremap_height, featuremap_width, 2]
        grid_scale_wh = torch.empty(grid_scale_wh_shape).fill_(area)  # shape=[num_anchors, featuremap_dim, featuremap_dim, 2]
        scale = torch.Tensor(self.scales).unsqueeze(-1).unsqueeze(-1)
        grid_scale_wh[..., 0] = grid_scale_wh[..., 0] * scale # wh = scale * wh
        grid_scale_wh[..., 1] = grid_scale_wh[..., 1] * scale  # wh = scale * wh

        grid_wh = torch.cat([grid_wh, grid_scale_wh], dim=0)
        anchor = torch.cat([grid_yx, grid_wh], dim=-1)
        anchor = anchor.reshape([-1, 4])

        return anchor