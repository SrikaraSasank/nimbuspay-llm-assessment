"""Deterministic TRAIN-ONLY audit. Never imported by inference."""
import argparse, collections, copy, csv, datetime as dt, hashlib, json, re
from pathlib import Path
from decimal import Decimal
KEYS=['category','priority','amount_paise','txn_date','txn_id','channel','language','needs_human']
CATS={'payment_failed','refund','fraud','account_access','kyc','offers','other'}
def load(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,rows): Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def body(text): return '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('>'))
def repairs(y,text):
    y=copy.deepcopy(y); reasons=[]; t=body(text)
    def put(k,v,why):
        if k not in y or type(y[k]) is not type(v) or y[k]!=v:
            reasons.append(f'{why}: {k} {y.get(k)!r} -> {v!r}');y[k]=v
    if 'prio' in y and 'priority' not in y:
        old=y.pop('prio');put('priority',{'low':'P3','medium':'P2','high':'P1'}.get(old,old),'legacy priority key/value')
    if y.get('category') in {'account_access','kyc','other'}:
        for k in ['amount_paise','txn_date','txn_id','channel']:put(k,None,'nontransaction category')
    else:
        if isinstance(y.get('txn_date'),str) and re.fullmatch(r'\d{2}/\d{2}/\d{4}',y['txn_date']):
            try:put('txn_date',dt.datetime.strptime(y['txn_date'],'%d/%m/%Y').date().isoformat(),'legacy date format')
            except ValueError:pass
        if isinstance(y.get('txn_id'),str):
            normalized=re.sub('[^A-Z0-9]','',y['txn_id'].upper())
            if re.fullmatch('NP[0-9]{10}',normalized):put('txn_id',normalized,'normalize transaction ID')
        # Only explicit transaction-amount templates; never infer from bare numbers.
        number=r'(\d[\d,]*(?:\.\d+)?)\s*(lakhs?\b|k\b)?'
        currency=r'(?:₹|INR\s*|Rs\.?\s*)?'
        pats=[r'\b(?:the transaction amount is|the amount was|amount\s*:?|it was for|it is|total)\s*'+currency+number,
              currency+number+r'\s*(?:rupees|rupaye)?\s*(?:kat gaye|was the value|is the amount|in total)']
        vals=set()
        for pat in pats:
            for m in re.finditer(pat,t,re.I):
                prefix=t[max(0,m.start()-45):m.start()].lower()
                if re.search(r'(balance|limit|fee)[^.!?\n]*$',prefix):continue
                v=Decimal(m[1].replace(',',''))*{'k':1000,'lakh':100000,'lakhs':100000,None:1}[m[2].lower() if m[2] else None]*100
                if v==v.to_integral_value():vals.add(int(v))
        if len(vals)==1:put('amount_paise',next(iter(vals)),'unique explicit claim amount')
        received=re.search(r'Received: (\d{4}-\d{2}-\d{2})',t)
        rel=[]
        # Longer expressions first so day-before-yesterday is not parsed twice.
        pat=r'day before yesterday|yesterday|\btoday\b|\bparso\b|\bkal\b|(\d+)\s+(?:days ago|din pehle)'
        for m in re.finditer(pat,t,re.I):
            rel.append(int(m[1]) if m[1] else {'day before yesterday':2,'yesterday':1,'today':0,'parso':2,'kal':1}[m[0].lower()])
        if received and len(set(rel))==1:
            d=dt.date.fromisoformat(received[1])-dt.timedelta(days=rel[0]);put('txn_date',d.isoformat(),'unambiguous relative date')
    amount=y.get('amount_paise');cat=y.get('category')
    legal=bool(re.search(r'\b(?:RBI|ombudsman|consumer court|lawyer|legal notice|legal action)\b',t,re.I))
    if cat in CATS and (amount is None or type(amount) is int):
        p='P1' if cat=='fraud' or (amount or 0)>=5000000 or legal else 'P2' if cat=='account_access' or cat in {'payment_failed','refund'} and (amount or 0)>=500000 else 'P3'
        put('priority',p,'schema priority precedence')
        repeat=bool(re.search(r'third time|second complaint|already written \d+ times|again and again|baar baar|pehle bhi complaint|pehle bhi support',t,re.I))
        if p=='P1' or repeat:put('needs_human',True,'mandatory escalation')
    return y,reasons

def valid(y):
    if not isinstance(y,dict) or set(y)!=set(KEYS):return False
    if y['category'] not in CATS or y['priority'] not in {'P1','P2','P3'} or y['language'] not in {'en','hi-en'}:return False
    if type(y['needs_human']) is not bool or y['amount_paise'] is not None and type(y['amount_paise']) is not int:return False
    if y['channel'] not in {None,'upi','card','netbanking','wallet'}:return False
    if y['txn_id'] is not None and (not isinstance(y['txn_id'],str) or not re.fullmatch(r'NP\d{10}',y['txn_id'])):return False
    if y['txn_date'] is not None:
        try:
            if dt.date.fromisoformat(y['txn_date']).isoformat()!=y['txn_date']:return False
        except (ValueError,TypeError):return False
    return True

def audit(source,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=load(source)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['messages'][1]['content']].append(r)
    conflicts={t for t,rr in groups.items() if len({r['messages'][-1]['content'] for r in rr})>1}
    seen={};clean=[];logs=[];changes=[];counts=collections.Counter();flags=[]
    for r in rows:
        text=r['messages'][1]['content'];why=[];rr=copy.deepcopy(r);action='fix'
        if text in conflicts:action='drop';why=['conflicting duplicate input: no majority-label guess']
        elif text in seen:action='drop';why=[f'exact duplicate of {seen[text]}']
        else:
            seen[text]=r['id']
            try:y=json.loads(r['messages'][-1]['content'])
            except json.JSONDecodeError:y=None
            if not isinstance(y,dict):action='drop';why=['invalid/truncated target JSON; no label reconstruction']
            else:
                fixed,why=repairs(y,text)
                if not valid(fixed):action='drop';why.append('unresolved schema violation')
                else:
                    rr['messages'][-1]['content']=json.dumps(fixed,ensure_ascii=False)
                    # Training-only removal. Inference always gets ORIGINAL complete input.
                    if body(text)!=text:
                        rr['messages'][1]['content']=body(text).strip();why.append('remove quoted earlier emails from training input')
                    if y!=fixed:changes.append({'id':r['id'],'before':y,'after':fixed,'evidence':text,'reasons':why.copy()})
                    if len(text)>1000 or y.get('category')=='fraud':flags.append({'id':r['id'],'reason':'manual semantic review recommended; automated rules are incomplete'})
        if why:
            logs.append({'id':r['id'],'action':action,'reason':'; '.join(why)})
            for w in why:counts[w.split(':')[0]]+=1
        if action!='drop':clean.append(rr)
    write(out/'train_clean.jsonl',clean);write(out/'audit_changes.jsonl',changes);write(out/'review_queue.jsonl',flags)
    with (out/'cleaning_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','action','reason']);w.writeheader();w.writerows(logs)
    summary={'input_rows':len(rows),'kept_rows':len(clean),'dropped_rows':sum(l['action']=='drop' for l in logs),'fixed_rows':sum(l['action']=='fix' for l in logs),'exact_duplicate_extras':sum(len(v)-1 for v in groups.values()),'whitespace_normalized_duplicate_extras':len(rows)-len({re.sub(r'\s+',' ',r['messages'][1]['content']).strip() for r in rows}), 'long_inputs_over_1000_chars':sum(len(r['messages'][1]['content'])>1000 for r in rows),'conflicting_input_groups':len(conflicts),'reasons':dict(counts),'train_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'limitations':'Conservative deterministic repairs; no semantic relabelling model, no dev/test access. Inspect audit_changes and review_queue. Remaining category, language, absolute-date and channel noise may persist.'}
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--out',default='private_data');a=p.parse_args();audit(a.train,a.out)
