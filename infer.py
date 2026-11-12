"""Generate raw predictions from the submitted adapter. Never repairs fields."""
import argparse,json
from pathlib import Path
from peft import PeftModel
from pipeline import base_model,tokenizer,predict
from score import load
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--adapter',default='adapter');p.add_argument('--config',default='best_config.json')
a=p.parse_args()
assert not Path(a.output).exists(),'Use a new output filename to force regeneration'
cfg=json.loads(Path(a.config).read_text())
model=PeftModel.from_pretrained(base_model(cfg),a.adapter)
predict(model,tokenizer(cfg),load(a.input),a.output)
