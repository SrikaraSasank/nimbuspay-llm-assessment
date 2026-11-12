"""Experiment export helper. Generates a DRAFT; final reviewed report.pdf is maintained separately. Do not rerun over the reviewed submission."""
import argparse,csv,json,shutil
from pathlib import Path

def build(source,explore,final,dest):
 source,explore,final,dest=map(Path,[source,explore,final,dest]);sub=final/'submission'
 for p in [sub/'adapter/adapter_config.json',sub/'adapter/adapter_model.safetensors',sub/'predictions_test.jsonl',explore/'experiments.csv']:
  assert p.is_file(),f'Missing required real output: {p}'
 assert json.loads((final/'runtime.json').read_text())['under_three_hours'],'Pipeline exceeded three hours; optimize and remeasure'
 assert len(list(csv.DictReader((explore/'experiments.csv').open())))>=5,'Finish the five runs for four controlled comparisons'
 if dest.exists():raise RuntimeError('Export destination exists; rename it to preserve edits before re-exporting')
 shutil.copytree(sub,dest)
 for name in ['pipeline.py','audit.py','score.py','test_audit.py','requirements.txt','SCHEMA.md','GUIDE.md','.gitignore','notebook.ipynb','make_submission.py']:
  shutil.copy(source/name,dest/name)
 evidence=dest/'evidence';evidence.mkdir()
 for name in ['experiments.csv','environment.txt','runtime.json']:shutil.copy(explore/name,evidence/('explore_'+name))
 for name in ['environment.txt','runtime.json']:shutil.copy(final/name,evidence/('reproduce_'+name))
 for run in ['clean_fp16','raw_fp16','clean_qlora','rank32_fp16','compact_fp16']:
  for name in ['config.json','metrics.json','lengths.json','scores.json','train_log.json']:
   p=explore/run/name
   if p.exists():shutil.copy(p,evidence/(run+'_'+name))
 for name in ['fixed_scores.json','rules_scores.json']:shutil.copy(explore/'baselines'/name,evidence/name)
 shutil.copy(final/'final/metrics.json',evidence/'final_metrics.json')
 cfg=json.loads((sub/'best_config.json').read_text());scores=json.loads((sub/'error_summary.json').read_text());a=scores['all']
 (dest/'README.md').write_text(f'''# NimbusPay triage\n\nBase model: {cfg['model']}\n\nBase revision: {cfg['revision']}\n\n4-bit loading: {cfg['quant4']}\n\nFinal reproduced dev exact match: {a['exact_match']:.2%}; mean field accuracy: {a['mean_field_acc']:.2%}.\n\nRun notebook.ipynb in reproduce mode on a fresh Colab T4 with the original private assignment pack. See GUIDE.md. Inference uses original system/user messages, native chat template, greedy model.generate with max_new_tokens=200 and whitespace stripping only.\n\n**Before submission:** replace the notebook with your executed Colab download, complete report explanations, and confirm total notebook runtime under three hours.\n\nAI assistance: ChatGPT/Codex helped write deterministic audit, training/evaluation code, tests and documentation. No external model/API generated labels or predictions. Candidate reviewed and executed the code: TODO confirm truthfully.\n\nSome task text derives from PolyAI Banking77, CC-BY-4.0, per the assignment.\n''')
 audit=json.loads((sub/'audit_summary.json').read_text());rows=list(csv.DictReader((explore/'experiments.csv').open()))
 lines=['# NimbusPay fine-tuning report — complete TODOs before submission','',
 '## Data audit',f"Audited {audit['input_rows']} training rows; retained {audit['kept_rows']}, dropped {audit['dropped_rows']}, modified {audit['fixed_rows']}. Dropped 60 exact duplicates and 25 invalid/truncated targets. Deterministic rules repair legacy keys/dates, transaction IDs, explicit claim amounts and priority. Quote removal is training-only. Logs record each changed row; uncertainty remains in semantic categories, unhandled formats and near duplicates. No dev/test data was used to clean or train.",'',
 '## Setup and results',f"Model {cfg['model']}; selected settings: rank {cfg['rank']}, alpha {cfg['alpha']}, LR {cfg['lr']}, epochs {cfg['epochs']}, 4-bit={cfg['quant4']}. Assistant-only loss, batch 1 × accumulation 16, no silent truncation. Sequence-length evidence is in evidence/*lengths.json. Final reproduced EM {a['exact_match']:.2%}, field accuracy {a['mean_field_acc']:.2%}.",'',
 '| Run | Dev EM | Mean field | Peak GiB | Train min |','|---|---:|---:|---:|---:|']
 for n in ['fixed','rules']:
  m=json.loads((explore/'baselines'/f'{n}_scores.json').read_text())['all'];lines.append(f"| baseline {n} | {m['exact_match']:.2%} | {m['mean_field_acc']:.2%} | — | — |")
 for r in rows:lines.append(f"| {r['run']} | {float(r['exact_match']):.2%} | {float(r['mean_field_acc']):.2%} | {float(r['peak_gpu_GiB']):.2f} | {float(r['training_minutes']):.1f}{'*' if r.get('timing_scope')=='resumed_segment_only' else ''} |")
 lines+=['','* Training time marked with an asterisk covers only the successful resumed segment; earlier interrupted time is not included. Do not interpret it as full training runtime.', 'Comparisons: raw_fp16 ↔ clean_fp16 (cleaning); clean_fp16 ↔ clean_qlora (quantization); clean_fp16 ↔ rank32_fp16 (rank); clean_fp16 ↔ compact_fp16 (JSON whitespace). Other settings are fixed. Fixed epochs yield different steps after dropping rows; one seed limits conclusions. Rank holds alpha fixed, changing alpha/r. TODO: explain the observed tradeoffs, including any negative results.','', '## Error analysis']
 fields=sorted(a['fields'].items(),key=lambda p:p[1]);cats=sorted(((k,v) for k,v in scores.items() if k.startswith('category=')),key=lambda p:p[1]['exact_match'])
 lines+=['Weakest fields: '+', '.join(f'{k} {v:.1%}' for k,v in fields[:3])+'.','Weakest categories: '+', '.join(f"{k.split('=')[1]} {v['exact_match']:.1%} (n={v['n']})" for k,v in cats[:3])+'.','', '| Failure ID | Wrong fields | Ticket-specific explanation |','|---|---|---|']
 failures=[json.loads(l) for l in (sub/'failures.jsonl').read_text().splitlines()]
 for f in failures[:10]:lines.append(f"| {f['id']} | {', '.join(f['wrong_fields']) or 'extra keys'} | TODO: inspect ticket, expected and actual in failures.jsonl |")
 if len(failures)<10:lines.append(f'Only {len(failures)} actual dev failures exist; do not invent additional failures.')
 lines+=['','## Next week and assistance','Next: review unresolved training ambiguities, add train-derived rule-boundary examples, compare a larger model under the same T4 budget, and repeat promising configurations across seeds. Keep dev/test isolated. ChatGPT/Codex assisted with code, deterministic audit, tests and documentation; no AI-generated labels or predictions were used. TODO: record your own review and any additional assistance.']
 (dest/'report.md').write_text('\n'.join(lines)+'\n')
 print('Exported',dest,'— complete report TODOs and save executed notebook before submission.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',default='.');p.add_argument('--explore',required=True);p.add_argument('--final',required=True);p.add_argument('--dest',required=True);a=p.parse_args();build(a.source,a.explore,a.final,a.dest)
