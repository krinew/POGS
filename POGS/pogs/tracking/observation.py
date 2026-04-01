import torch
from typing import List, Optional, Callable, Generic, TypeVar
from nerfstudio.cameras.cameras import Cameras
from torchvision.transforms.functional import resize
from pogs.tracking.utils import *
from pogs.tracking.utils2 import *
from copy import deepcopy

T = TypeVar('T')
class Future(Generic[T]):
    """
    A simple wrapper for deferred execution of a callable until retrieved
    """
    def __init__(self,callable):
        self.callable = callable
        self.executed = False

    def retrieve(self):
        if not self.executed:
            self.result = self.callable()
            self.executed = True
        return self.result
    
class Frame:
    rasterize_resolution: int = 500
    camera: Cameras
    rgb: torch.Tensor
    metric_depth: bool
    _depth: Future[torch.Tensor]
    _dino_feats: Future[torch.Tensor]
    # _hand_mask: Future[torch.Tensor]

    @property
    def depth(self):
        return self._depth.retrieve()
    
    @property
    def dino_feats(self):
        return self._dino_feats.retrieve()

    # @property
    # def hand_mask(self):
    #     return self._hand_mask.retrieve()
    
    @property
    def mask(self):
        return self._mask.retrieve()
    
    def __init__(self, rgb: torch.Tensor, camera: Cameras, dino_fn: Callable, metric_depth_img: Optional[torch.Tensor], 
                 xmin: Optional[float] = None, xmax: Optional[float] = None, ymin: Optional[float] = None, ymax: Optional[float] = None):

        self.camera = deepcopy(camera.to('cuda'))

        self._dino_fn = dino_fn
        self.rgb = resize(
                rgb.permute(2, 0, 1),
                (camera.height, camera.width),
                antialias=True,
            ).permute(1, 2, 0)
        self.metric_depth = metric_depth_img is not None
        self.obj_mask = None     
        
           
        @torch.no_grad()
        def _get_depth():
            if metric_depth_img is not None:
                depth = metric_depth_img
            else:
                raise FileNotFoundError
            depth = resize(
                            depth.unsqueeze(0),
                            (camera.height, camera.width),
                            antialias=True,
                        ).squeeze().unsqueeze(-1)
            return depth
        self._depth = Future(_get_depth)
        @torch.no_grad()
        def _get_dino():
            dino_feats = dino_fn(
                rgb.permute(2, 0, 1).unsqueeze(0)
            ).squeeze()
            dino_feats = resize(
                dino_feats.permute(2, 0, 1),
                (camera.height, camera.width),
                antialias=True,
            ).permute(1, 2, 0)
            return dino_feats
        self._dino_feats = Future(_get_dino)
        # @torch.no_grad()
        # def _get_hand_mask():
        #     hand_mask = get_hand_mask((self.rgb * 255).to(torch.uint8))
        #     hand_mask = (
        #         torch.nn.functional.max_pool2d(
        #             hand_mask[None, None], 3, padding=1, stride=1
        #         ).squeeze()
        #         == 0.0
        #     )
        #     return hand_mask
        # self._hand_mask = Future(_get_hand_mask)
        @torch.no_grad()
        def _get_mask():
            obj_mask = resize(
                            self.obj_mask.unsqueeze(0),
                            (camera.height, camera.width),
                            antialias=True,
                        ).squeeze(0)
            return obj_mask
        self._mask = Future(_get_mask)
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        

    
class PosedObservation:
    """
    Class for computing relevant data products for a frame and storing them
    """
    max_roi_resolution: int = 490
    _frame: Frame
    _raw_rgb: torch.Tensor
    _original_camera: Cameras
    _original_depth: Optional[torch.Tensor] = None
    _roi_frames: Optional[List[Frame]] = None
    
    def __init__(self, rgb: torch.Tensor, camera: Cameras, dino_fn: Callable, metric_depth_img: Optional[torch.Tensor] = None):
        """
        Initialize the frame

        rgb: HxWx3 tensor of the rgb image, normalized to [0,1]
        camera: Cameras object for the camera intrinsics and extrisics to render the frame at
        dino_fn: callable taking in 3HxW RGB image and outputting dino features CxHxW
        metric_depth_img: HxWx1 tensor of metric depth, if desired
        """
        assert rgb.shape[0] == camera.height and rgb.shape[1] == camera.width, f"Input image should be the same size as the camera, got {rgb.shape} and {camera.height}x{camera.width}"
        self._dino_fn = dino_fn
        assert rgb.shape[-1] == 3, rgb.shape
        self._raw_rgb = rgb
        if metric_depth_img is not None:
            self._original_depth = metric_depth_img
        self._original_camera = deepcopy(camera.to('cuda'))
        cam = deepcopy(camera.to('cuda'))

        self._frame = Frame(rgb, cam, dino_fn, metric_depth_img)
        self._roi_frames = []
        self._obj_masks = None
        
        
    @property
    def frame(self):
        return self._frame
    
    @property
    def roi_frames(self):
        if len(self._roi_frames) == 0:
            raise RuntimeError("ROIs not set")
        return self._roi_frames
    
    def add_roi(self, xmin, xmax, ymin, ymax):
        assert xmin < xmax and ymin < ymax
        assert xmin >= 0 and ymin >= 0
        assert xmax <= 1.0 and ymax <= 1.0, "xmin and ymin should be normalized"
        # convert normalized to pixels in original image
        xmin,xmax,ymin,ymax = int(xmin*(self._original_camera.width-1)), int(xmax*(self._original_camera.width-1)),\
              int(ymin*(self._original_camera.height-1)), int(ymax*(self._original_camera.height-1))
        # adjust these value to be multiples of 14, dino patch size
        xlen = ((xmax - xmin)//14) * 14
        ylen = ((ymax - ymin)//14) * 14
        xmax = xmin + xlen
        ymax = ymin + ylen
        rgb = self._raw_rgb[ymin:ymax, xmin:xmax].clone()
        camera = crop_camera(self._original_camera, xmin, xmax, ymin, ymax)
        if max(camera.width.item(),camera.height.item()) > self.max_roi_resolution:
            camera.rescale_output_resolution(self.max_roi_resolution/max(camera.width.item(),camera.height.item()))
        depth = self._original_depth[ymin:ymax, xmin:xmax].clone().squeeze(-1)
        self._roi_frames.append(Frame(rgb, camera, self._dino_fn, depth, xmin, xmax, ymin, ymax))
        
    def update_roi(self, idx, xmin, xmax, ymin, ymax):
        assert len(self._roi_frames) > idx
        assert xmin < xmax and ymin < ymax
        assert xmin >= 0 and ymin >= 0
        assert xmax <= 1.0 and ymax <= 1.0, "xmin and ymin should be normalized"
        # convert normalized to pixels in original image
        xmin,xmax,ymin,ymax = int(xmin*(self._original_camera.width-1)), int(xmax*(self._original_camera.width-1)),\
              int(ymin*(self._original_camera.height-1)), int(ymax*(self._original_camera.height-1))
        # adjust these value to be multiples of 14, dino patch size
        xlen = ((xmax - xmin)//14) * 14
        ylen = ((ymax - ymin)//14) * 14
        xmax = xmin + xlen
        ymax = ymin + ylen
        rgb = self._raw_rgb[ymin:ymax, xmin:xmax].clone()
        camera = crop_camera(self._original_camera, xmin, xmax, ymin, ymax)
        if max(camera.width.item(),camera.height.item()) > self.max_roi_resolution:
            camera.rescale_output_resolution(self.max_roi_resolution/max(camera.width.item(),camera.height.item()))
        depth = self._original_depth[ymin:ymax, xmin:xmax].clone().squeeze(-1)

        self._roi_frames[idx] = Frame(rgb, camera, self._dino_fn, depth, xmin, xmax, ymin, ymax)
        if len(self._obj_masks) > 0:

            self._roi_frames[idx].obj_mask = self._obj_masks[idx].squeeze(0)[ymin:ymax, xmin:xmax].clone()