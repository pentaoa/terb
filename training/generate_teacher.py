"""Generate Beat This! small0 teacher probabilities for prepared training files."""
import argparse,collections,inspect,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/beat_this-reference'))
from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key
def main():
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('checkpoint');p.add_argument('--batch',type=int,default=4);p.add_argument('--skip-existing',action='store_true');a=p.parse_args()
 rows=[json.loads(x) for x in open(a.manifest)]
 if a.skip_existing: rows=[r for r in rows if 'teacher_beat' not in np.load(r['feature']).files]
 groups=collections.defaultdict(list)
 for r in rows:groups[np.load(r['feature'])['mel'].shape[0]].append(r)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);hp={k:v for k,v in ck['hyper_parameters'].items() if k in set(inspect.signature(BeatThis).parameters)};device='cuda' if torch.cuda.is_available() else 'cpu';m=BeatThis(**hp).eval().to(device);m.load_state_dict(replace_state_dict_key(ck['state_dict'],'model.',''))
 with torch.inference_mode():
  for _,items in sorted(groups.items()):
   for start in range(0,len(items),a.batch):
    batch=items[start:start+a.batch];loaded=[np.load(r['feature']) for r in batch];x=torch.from_numpy(np.stack([z['mel'].astype('float32') for z in loaded])).to(device);pred=m(x)
    for r,z,b,d in zip(batch,loaded,pred['beat'],pred['downbeat']):
     arrays={k:z[k] for k in z.files};arrays['teacher_beat']=torch.sigmoid(b).cpu().numpy().astype('float16');arrays['teacher_downbeat']=torch.sigmoid(d).cpu().numpy().astype('float16');np.savez(r['feature'],**arrays)
 print(json.dumps({'recordings':len(rows),'teacher':'Beat This! small0','device':device}))
if __name__=='__main__':main()
