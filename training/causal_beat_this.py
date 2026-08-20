"""Turn a Beat This! model into a bounded-lookahead model for fine-tuning.

The adapter masks every attention operation that runs along time. Frequency
attention remains bidirectional because it does not mix frames. Four symmetric
3-frame convolutions retain exactly four frames (80 ms at 50 fps) of lookahead.
The resulting model still needs fine-tuning; enabling masks is not by itself a
production conversion.
"""
from pathlib import Path
import inspect
import sys
import torch
import torch.nn.functional as F


LOOKAHEAD_FRAMES = 4


def load_reference(reference_root):
    root=str(Path(reference_root).resolve())
    if root not in sys.path: sys.path.insert(0,root)
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.model import roformer
    from beat_this.utils import replace_state_dict_key
    return BeatThis,roformer,replace_state_dict_key


def enable_causal_time_attention(model,roformer,context_frames=None):
    def attend(self,q,k,v):
        if self.scale is not None:
            default_scale=q.shape[-1]**-0.5
            q=q*(self.scale/default_scale)
        causal=bool(getattr(self,"causal",False))
        context=getattr(self,"context_frames",None)
        if causal and context and q.shape[-2]>context:
            n=q.shape[-2]; qi=torch.arange(n,device=q.device)[:,None]; ki=torch.arange(n,device=q.device)[None,:]
            allowed=(ki<=qi)&(ki>qi-context)
            return F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=self.dropout if self.training else 0.0)
        return F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0,is_causal=causal)
    roformer.Attend.forward=attend

    count=0
    for block in model.frontend.blocks:
        partial=block.partial
        if hasattr(partial,"attnT"):
            partial.attnT.attend.causal=True
            partial.attnT.attend.context_frames=context_frames
            count+=1
        if hasattr(partial,"attnF"):
            partial.attnF.attend.causal=False
    for attn,_ff in model.transformer_blocks.layers:
        attn.attend.causal=True
        attn.attend.context_frames=context_frames
        count+=1
    if count==0:
        raise ValueError("model has no time-attention modules to mask")
    return model


def load_causal_checkpoint(reference_root,checkpoint,device="cpu",context_frames=None):
    BeatThis,roformer,replace_key=load_reference(reference_root)
    ck=torch.load(checkpoint,map_location=device,weights_only=True)
    valid=set(inspect.signature(BeatThis).parameters)
    hp={k:v for k,v in ck["hyper_parameters"].items() if k in valid}
    model=BeatThis(**hp)
    model.load_state_dict(replace_key(ck["state_dict"],"model.",""))
    return enable_causal_time_attention(model,roformer,context_frames).to(device)
