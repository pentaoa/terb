"""Fail closed unless a candidate meets the production tempo and runtime gates."""
import argparse,json,sys
p=argparse.ArgumentParser();p.add_argument('metrics');p.add_argument('model_card');a=p.parse_args()
m=json.load(open(a.metrics))['summary'];c=json.load(open(a.model_card));checks={
 'strict_accuracy>=0.70':m['strict_accuracy']>=.70,
 'metrical_accuracy>=0.85':m['metrical_accuracy']>=.85,
 'half_time_rate<=0.10':m['half_time_rate']<=.10,
 'wrong_rate<=0.15':m['wrong_rate']<=.15,
 'accuracy_4s>=0.55':m['accuracy_4s']>=.55,
 'cpu_rtf<=0.25':c['cpu_smoke']['realtime_factor']<=.25,
 'model_size<=15MiB':c['model_bytes']<=15*1024*1024,
}
print(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2));sys.exit(0 if all(checks.values()) else 1)
