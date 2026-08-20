"""Export a bounded-lookahead Beat This! checkpoint to static opset-17 ONNX."""
import argparse,json
from pathlib import Path
import torch
from causal_beat_this import load_causal_checkpoint


class Outputs(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,x):
        pred=self.model(x)
        return pred["beat"],pred["downbeat"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--reference-root",default="data/beat_this-reference")
    p.add_argument("--frames",type=int,default=800)
    p.add_argument("--attention-context",type=int,default=84)
    a=p.parse_args()
    model=Outputs(load_causal_checkpoint(
        a.reference_root,a.checkpoint,"cpu",a.attention_context
    ).eval())
    dummy=torch.zeros(1,a.frames,128)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,dummy,output,
            input_names=["spectrogram"],
            output_names=["beat","downbeat"],
            opset_version=17,
            dynamic_axes=None,
            do_constant_folding=True,
        )
    import onnx
    graph=onnx.load(str(output)); onnx.checker.check_model(graph)
    meta={
        "model":"bounded-causal-small-transformer-v1",
        "source_checkpoint":str(Path(a.checkpoint)),
        "input_shape":[1,a.frames,128],
        "attention_context_per_layer":a.attention_context,
        "past_receptive_frames":751,
        "future_lookahead_frames":4,
        "parameters":sum(p.numel() for p in model.parameters()),
        "opset":17,
        "model_bytes":output.stat().st_size,
    }
    output.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))


if __name__=="__main__":
    main()
