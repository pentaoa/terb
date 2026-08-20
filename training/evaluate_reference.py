"""Fair Beat This! small0 reference on GTZAN spectrograms, no DBN."""
import argparse,collections,inspect,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/beat_this-reference'))
from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key
from evaluate import peaks,f1,reference,category
from evaluate_production_decoder import tempo

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('checkpoint');p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=4);a=p.parse_args()
 rows=[json.loads(x) for x in open(a.manifest)];groups=collections.defaultdict(list)
 for r in rows:
  z=np.load(r['feature']);groups[z['mel'].shape[0]].append(r)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);valid=set(inspect.signature(BeatThis).parameters);hp={k:v for k,v in ck['hyper_parameters'].items() if k in valid};m=BeatThis(**hp).eval();m.load_state_dict(replace_state_dict_key(ck['state_dict'],'model.',''));torch.set_num_threads(max(1,torch.get_num_threads()));out=[];started=time.perf_counter()
 with torch.inference_mode():
  for length,items in sorted(groups.items()):
   for start in range(0,len(items),a.batch):
    batch=items[start:start+a.batch];zs=[np.load(r['feature']) for r in batch];x=torch.from_numpy(np.stack([z['mel'].astype('float32') for z in zs]));pred=m(x)
    for r,z,beat,down in zip(batch,zs,pred['beat'].numpy(),pred['downbeat'].numpy()):
     # Official outputs logits; sigmoid only for F1 threshold. Tempo is scale-invariant after centering.
     act=1/(1+np.exp(-beat));ref=reference(z['beat']);item={'recording_id':r['recording_id'],'reference_bpm':ref,'beat_f1':f1(peaks(act,.5),peaks(z['beat'],.9))}
     if 'downbeat' in z:item['downbeat_f1']=f1(peaks(1/(1+np.exp(-down)),.5),peaks(z['downbeat'],.9))
     for sec in (4,8,16,30):item[f'bpm_{sec}s']=tempo(act,sec)
     item['bpm_final']=tempo(act);item['classification']=category(item['bpm_final'],ref);out.append(item)
 n=len(out);rate=lambda k:sum(x['classification']==k for x in out)/max(1,n);summary={'model':'Beat This! small0','recordings':n,'seconds':time.perf_counter()-started,'beat_f1':float(np.mean([x['beat_f1'] for x in out])),'downbeat_f1':float(np.mean([x.get('downbeat_f1',0) for x in out])),'strict_accuracy':rate('correct'),'metrical_accuracy':sum(rate(k) for k in ('correct','half_time','double_time')),'half_time_rate':rate('half_time'),'double_time_rate':rate('double_time'),'wrong_rate':rate('wrong')}
 for sec in (4,8,16,30):summary[f'accuracy_{sec}s']=sum(category(x[f'bpm_{sec}s'],x['reference_bpm'])=='correct' for x in out)/max(1,n)
 Path(a.out).write_text(json.dumps({'summary':summary,'recordings':out},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
