"""Numeric body evidence adapter used by the coarse framing decoder."""
import torch
from torch import nn
from .framing import FramingDecoder, decode_geometry
from .constants import FRAMING, PARTS, BOUNDARY, VISIBILITY

class OnlineDecoder(FramingDecoder):
    def __init__(self,c,language_dim):
        super().__init__(c,language_dim);d=c['hidden_dim']
        del self.semantic
        self.body_evidence=nn.Linear(c['body_dim'],d)
        self.fuse=nn.Sequential(nn.Linear(5*d,d),nn.GELU(),nn.LayerNorm(d))
        # Do not train arbitrary body categories through geometry while their
        # semantic supervision is pending; unknown output has explicit authority.
        self.body_head.requires_grad_(False);self.body_condition.requires_grad_(False)

    def forward(self,regional,global_h,source_h,g,relations):
        n=regional.shape[1];gh=self.global_readout(global_h.float())
        x=self.fuse(torch.cat([self.region_readout(regional.float()),gh[:,None].expand(-1,n,-1),
            source_h,self.geometry(g['geometry']),self.body_evidence(g['body'])],-1))
        for block in self.blocks:x=block(x,g['node_mask'],relations)
        emphasis=(self.focus(x).squeeze(-1)+4*g['focus_hint']).masked_fill(~g['node_mask'],-torch.inf).softmax(-1)
        whole=self.global_fuse(torch.cat([gh,(x*emphasis[...,None]).sum(1)],-1))
        log_aspect=g['geometry'][:,0,12]+self.frame_head(whole).squeeze(-1)
        x=x+whole[:,None]+self.frame_condition(log_aspect[:,None])[:,None]
        x=self.final_block(x,g['node_mask'],relations)
        boxes,centers,areas=decode_geometry(self.geometry_head(x),g['geometry'])
        def unknown(classes,parts=False):
            shape=(*x.shape[:2],len(PARTS),len(classes)) if parts else (*x.shape[:2],len(classes))
            value=x.new_full(shape,-10000.);value[...,classes.index('unknown')]=0.;return value
        return {'log_aspect':log_aspect,'boxes':boxes,'centroids':centers,'areas':areas,
                'framing_logits':unknown(FRAMING),'boundary_logits':unknown(BOUNDARY,True),
                'visibility_logits':unknown(VISIBILITY,True),'emphasis':emphasis,
                'emphasis_semantics':'internal_learned_allocation_not_calibrated_subject_truth',
                'body_guidance_enabled':False}
