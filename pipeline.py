"""Colab T4 training and raw greedy inference; no output repair."""
import argparse, copy, csv, gc, hashlib, json, math, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import model_info
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, set_seed, GenerationConfig)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from score import load, score, parse, same, KEYS
MODEL='Qwen/Qwen2.5-0.5B-Instruct'
BASE=dict(model=MODEL,quant4=False,rank=16,alpha=32,lr=2e-4,epochs=2,seed=42,clean=True,compact=False)
EXPERIMENTS={
 'clean_fp16':dict(BASE),
 'raw_fp16':dict(BASE,clean=False),
 'clean_qlora':dict(BASE,quant4=True),
 'rank32_fp16':dict(BASE,rank=32),
 'compact_fp16':dict(BASE,compact=True),
}

def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def jsonl(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def tokenizer(cfg):
    t=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision']);t.pad_token=t.eos_token;return t

def base_model(cfg):
    kw=dict(revision=cfg['revision'],torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
    if cfg['quant4']:kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(cfg['model'],**kw)

def encode_row(row,t,compact=False):
    msgs=copy.deepcopy(row['messages'])
    if compact:msgs[-1]['content']=json.dumps(json.loads(msgs[-1]['content']),ensure_ascii=False,separators=(',',':'))
    prompt=t.apply_chat_template(msgs[:2],tokenize=True,add_generation_prompt=True)
    full=t.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False)
    assert full[:len(prompt)]==prompt,'Chat-template boundary mismatch'
    labels=[-100]*len(prompt)+full[len(prompt):]
    assert any(v!=-100 for v in labels)
    return dict(input_ids=full,attention_mask=[1]*len(full),labels=labels)

class Collator:
    def __init__(self,t):self.t=t
    def __call__(self,examples):
        n=math.ceil(max(len(e['input_ids']) for e in examples)/8)*8
        return {k:torch.tensor([e[k]+[(-100 if k=='labels' else self.t.pad_token_id if k=='input_ids' else 0)]*(n-len(e[k])) for e in examples]) for k in ['input_ids','attention_mask','labels']}

def predict(model,t,rows,path,rules=None):
    """Only original system/user messages, native template, greedy generation."""
    model.eval();model.config.use_cache=True
    gen=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=200,
                         eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id)
    result=[]
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try: item=json.loads(line)
            except json.JSONDecodeError: break
            assert len(result)<len(rows) and item['id']==rows[len(result)]['id'], 'Prediction IDs changed'
            assert isinstance(item.get('output'),str)
            result.append(item)
        jsonl(path,result)
    print(f'{Path(path).name}: {len(result)}/{len(rows)} saved predictions',flush=True)
    with open(path,'a') as f:
        for i in range(len(result),len(rows)):
            r=rows[i]
            msgs=copy.deepcopy(r['messages'][:2])
            assert [m['role'] for m in msgs]==['system','user']
            if rules is not None:msgs[0]['content']+='\n\n'+rules
            inputs=t.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
            with torch.inference_mode():out=model.generate(**inputs,generation_config=gen)
            raw=t.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            item={'id':r['id'],'output':raw};result.append(item);f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
            if (i+1)%10==0:print(f'Generated {i+1}/{len(rows)}',flush=True)
    return result

def train_one(name,cfg,pack,private,out):
    set_seed(cfg['seed']);dest=out/name
    if (dest/'metrics.json').exists():
        previous=json.loads((dest/'metrics.json').read_text())
        assert previous['config']==cfg,'Config changed; use a fresh output folder'
        print('Using completed run:',name,flush=True);return previous
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'config.json').exists():
        assert json.loads((dest/'config.json').read_text())==cfg, 'Settings changed: use a new run folder'
    dump(dest/'config.json',cfg)
    print('STARTING:',name,flush=True)
    t=tokenizer(cfg)
    if not (dest/'training_complete.json').exists():
        rows=load(private/'train_clean.jsonl' if cfg['clean'] else pack/'data/train.jsonl')
        ds=[encode_row(r,t,cfg['compact']) for r in rows];lengths=[len(r['input_ids']) for r in ds]
        maximum=math.ceil(max(lengths)/32)*32
        assert maximum<=8192,'Inspect long inputs; refusing silent truncation'
        dump(dest/'lengths.json',{'rows':len(rows),'percentiles':dict(zip(['50','95','99','100'],map(float,np.percentile(lengths,[50,95,99,100])))),'max_seq_length':maximum,'truncated_rows':0})
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model=base_model(cfg)
        assert sum(p.numel() for p in model.parameters())<=4_000_000_000
        if cfg['quant4']:model=prepare_model_for_kbit_training(model)
        else:model.enable_input_require_grads()
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        ntrain=sum(p.numel() for p in model.parameters() if p.requires_grad);assert ntrain<=50_000_000
        args=TrainingArguments(output_dir=str(dest/'checkpoints'),num_train_epochs=cfg['epochs'],per_device_train_batch_size=1,gradient_accumulation_steps=16,
            learning_rate=cfg['lr'],lr_scheduler_type='cosine',warmup_ratio=.05,weight_decay=.01,fp16=True,
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',
            logging_steps=10,save_strategy='steps',save_steps=25,save_total_limit=2,report_to='none',seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=0)
        trainer=Trainer(model=model,args=args,train_dataset=ds,data_collator=Collator(t))
        from transformers.trainer_utils import get_last_checkpoint
        resume=get_last_checkpoint(str(dest/'checkpoints')) if (dest/'checkpoints').exists() else None
        timing_scope='resumed_segment_only' if resume else 'full_training'
        print('Training checkpoint:',resume or 'fresh run',flush=True)
        torch.cuda.synchronize();start=time.perf_counter();trainer.train(resume_from_checkpoint=resume);torch.cuda.synchronize();minutes=(time.perf_counter()-start)/60
        peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
        model.save_pretrained(dest/'adapter',safe_serialization=True);t.save_pretrained(dest/'adapter')
        dump(dest/'config.json',cfg);dump(dest/'train_log.json',trainer.state.log_history)
        dump(dest/'training_complete.json',dict(minutes=minutes,peak=peak,reserved=reserved,ntrain=ntrain,maximum=maximum,timing_scope=timing_scope))
        # Reload the saved artifact exactly as evaluation will do, before scoring.
        del trainer,model;gc.collect();torch.cuda.empty_cache()
    info=json.loads((dest/'training_complete.json').read_text())
    minutes,peak,reserved,ntrain,maximum=(info[k] for k in ['minutes','peak','reserved','ntrain','maximum'])
    timing_scope=info['timing_scope']
    print('Evaluating saved adapter:',name,flush=True)
    model=PeftModel.from_pretrained(base_model(cfg),dest/'adapter');pred=predict(model,t,load(pack/'data/dev.jsonl'),dest/'predictions_dev.jsonl')
    scores=score(load(pack/'data/dev.jsonl'),pred);dump(dest/'scores.json',scores)
    metrics={'run':name,'config':cfg,'exact_match':scores['all']['exact_match'],'mean_field_acc':scores['all']['mean_field_acc'],'peak_gpu_GiB':peak,'peak_reserved_GiB':reserved,'training_minutes':minutes,'timing_scope':timing_scope,'trainable_parameters':ntrain,'max_seq_length':maximum}
    dump(dest/'metrics.json',metrics);print(json.dumps(metrics,indent=2),flush=True)
    del model;gc.collect();torch.cuda.empty_cache();return metrics

def baselines(cfg,pack,out):
    dest=out/'baselines';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==cfg
    dump(dest/'config.json',cfg)
    if all((dest/f'{n}_scores.json').exists() for n in ['fixed','rules']):
        print('Reusing completed baselines',flush=True);return
    dest=out/'baselines';dest.mkdir(parents=True,exist_ok=True);t=tokenizer(cfg);model=base_model(cfg);gold=load(pack/'data/dev.jsonl')
    for name,rules in [('fixed',None),('rules',(pack/'SCHEMA.md').read_text())]:
        pred=predict(model,t,gold,dest/f'{name}.jsonl',rules);dump(dest/f'{name}_scores.json',score(gold,pred))
    dump(dest/'config.json',cfg);del model;gc.collect();torch.cuda.empty_cache()

def analysis_files(gold,pred,out):
    by={r['id']:r['output'] for r in pred};fail=[]
    hints={'category':'Inspect competing intents and fraud precedence.','priority':'Check threshold, legal threat and category precedence.','amount_paise':'Check disputed amount, distractors and rupees-to-paise conversion.','txn_date':'Check received date, relative dates, missing year and quoted emails.','txn_id':'Check NP normalization and unrelated references.','channel':'Check explicit mode versus inferred mode.','language':'Check Hinglish and misspellings.','needs_human':'Check P1 or explicit repeat contact.'}
    for r in gold:
        actual=by.get(r['id'],'');p=parse(actual);g=json.loads(r['messages'][-1]['content']);wrong=[k for k in KEYS if p is None or k not in p or not same(p[k],g[k])]
        if wrong or p is not None and set(p)!=set(KEYS):
            fail.append({'id':r['id'],'ticket':r['messages'][1]['content'],'gold':g,'raw_output':actual,'wrong_fields':wrong,'diagnostic_prompts':[hints[k] for k in wrong],'candidate_explanation':'TODO: inspect this specific failure; prompts are not a causal explanation.'})
    jsonl(out/'failures.jsonl',fail)
    dump(out/'error_summary.json',score(gold,pred))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['explore','reproduce'],required=True);ap.add_argument('--pack',required=True);ap.add_argument('--private',required=True);ap.add_argument('--out',required=True);ap.add_argument('--best-config',default='best_config.json');ap.add_argument('--runs',default=','.join(EXPERIMENTS));a=ap.parse_args()
    assert torch.cuda.is_available(),'Choose Colab T4 GPU first'
    assert 'T4' in torch.cuda.get_device_name(0),'Assignment requires T4'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pack,private,out=map(Path,[a.pack,a.private,a.out]);out.mkdir(parents=True,exist_ok=True)
    had_prior_work=any(out.glob('*/config.json'))
    started=time.perf_counter();(out/'environment.txt').write_text(subprocess.check_output(['pip','freeze'],text=True)+'\nGPU: '+torch.cuda.get_device_name(0))
    if a.mode=='explore':
        revision_file=out/'model_revision.json'
        revision=json.loads(revision_file.read_text())['revision'] if revision_file.exists() else model_info(MODEL).sha
        dump(revision_file,{'revision':revision})
        cfg=dict(BASE,revision=revision);baselines(cfg,pack,out)
        metrics=[train_one(n,dict(EXPERIMENTS[n],revision=revision),pack,private,out) for n in a.runs.split(',')]
        best=max(metrics,key=lambda m:(m['exact_match'],m['mean_field_acc']));chosen=out/best['run'];cfg=best['config'];dump(a.best_config,cfg)
        with (out/'experiments.csv').open('w',newline='') as f:
            fields=['run','exact_match','mean_field_acc','peak_gpu_GiB','training_minutes','timing_scope','trainable_parameters'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(metrics)
    else:
        cfg=json.loads(Path(a.best_config).read_text());train_one('final',cfg,pack,private,out);chosen=out/'final'
    # Both modes export a real adapter and its matching raw predictions.
    submission=out/'submission';submission.mkdir(exist_ok=True);shutil.copytree(chosen/'adapter',submission/'adapter',dirs_exist_ok=True)
    dump(submission/'best_config.json',cfg);t=tokenizer(cfg);model=PeftModel.from_pretrained(base_model(cfg),submission/'adapter')
    test=load(pack/'data/test_inputs.jsonl');pred=predict(model,t,test,submission/'predictions_test.jsonl')
    assert len(pred)==len(test) and [p['id'] for p in pred]==[r['id'] for r in test]
    # Same-model repeat check against file, without parsing or fixing output.
    check=predict(model,t,test[:5],out/'reload_check.jsonl');assert check==pred[:5]
    analysis_files(load(pack/'data/dev.jsonl'),load(chosen/'predictions_dev.jsonl'),submission)
    shutil.copy(private/'cleaning_log.csv',submission/'cleaning_log.csv');shutil.copy(private/'audit_summary.json',submission/'audit_summary.json')
    dump(out/'runtime.json',{'mode':a.mode,'total_minutes':(time.perf_counter()-started)/60,'resumed_or_reused':had_prior_work,'under_three_hours':(time.perf_counter()-started)<10800 and not had_prior_work})
    print('Finished. Review errors, write report, save notebook WITH outputs. Submission:',submission)
if __name__=='__main__':main()
