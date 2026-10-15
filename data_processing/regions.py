"""Pure crop and padding utilities from the regional descriptor pipeline."""
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FeatureConfig:
    long_edge: int = 448
    region_margin: float = .05
    context_margin: float = .20  # per side
    layers: tuple = (17, 23)
    patch_size: int = 16


def crop_box(mask, margin):
    yy, xx = np.nonzero(mask)
    if not len(xx): raise ValueError('Cannot crop an empty mask')
    x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    dx,dy=(x1-x0)*margin,(y1-y0)*margin
    return (max(0,math.floor(x0-dx)),max(0,math.floor(y0-dy)),
            min(mask.shape[1],math.ceil(x1+dx)),min(mask.shape[0],math.ceil(y1+dy)))


def letterbox(rgb, mask, box, cfg):
    crop = rgb.crop(box)
    x0,y0,x1,y1=box
    part = mask[y0:y1,x0:x1]
    scale = cfg.long_edge/max(crop.size)
    w,h=max(1,round(crop.width*scale)),max(1,round(crop.height*scale))
    pw,ph=math.ceil(w/cfg.patch_size)*cfg.patch_size,math.ceil(h/cfg.patch_size)*cfg.patch_size
    # Mean-valued padding; feature pooling explicitly excludes padding.
    arr = np.empty((ph,pw,3),np.float32);arr[:]=[.485,.456,.406]
    arr[:h,:w]=np.asarray(crop.resize((w,h),Image.Resampling.BICUBIC),np.float32)/255
    foreground=np.zeros((ph,pw),np.float32)
    foreground[:h,:w]=np.asarray(Image.fromarray(part.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
    valid=np.zeros((ph,pw),np.float32);valid[:h,:w]=1
    transform={'crop_xyxy':list(box),'resized_wh':[w,h],'padded_wh':[pw,ph],
               'scale_xy':[w/crop.width,h/crop.height],'pad_xy':[0,0]}
    return arr,foreground,valid,transform
