import json
import re

with open('notebook8d611b5bf2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell with training outputs
train_output_text = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            txt = ''.join(out.get('text', []))
            if 'training is FINISHED' in txt or 'iteration,' in txt:
                train_output_text = txt
                break

print(f"Captured training output length: {len(train_output_text)} characters")

iter_pattern = re.compile(
    r'(\d+)\s+iteration,\s+({.*?}),\s+BEST_EVAL_AUC:\s+([0-9.]+),\s+at\s+(\d+)\s+iters'
)

records = []
for line in train_output_text.splitlines():
    m = iter_pattern.search(line)
    if m:
        it = int(m.group(1))
        best_auc = float(m.group(3))
        best_at = int(m.group(4))
        # clean numpy types from dict string
        d_str = m.group(2).replace('np.float64(', '').replace(')', '')
        try:
            d = eval(d_str)
            records.append({
                'it': it,
                'auc': d.get('eval/AUC'),
                'f1': d.get('eval/f1'),
                'acc': d.get('eval/acc'),
                'recall': d.get('eval/recall'),
                'specificity': d.get('eval/specificity'),
                'loss': d.get('train/total_loss'),
                'best_auc': best_auc,
                'best_at': best_at
            })
        except Exception as e:
            pass

print(f"Total evaluation points captured: {len(records)}")

if records:
    print("\n" + "="*85)
    print(f"{'Iter':<6} | {'AUROC (%)':<10} | {'F1 (%)':<8} | {'ACC (%)':<8} | {'Recall/SEN':<11} | {'Specificity':<11} | {'Loss':<7}")
    print("="*85)
    # Print key sample points: every 400 iters
    for r in records:
        if (r['it'] + 1) % 400 == 0 or r['it'] == records[-1]['it'] or r['auc'] == max(x['auc'] for x in records):
            is_peak = " *** PEAK ***" if r['auc'] == max(x['auc'] for x in records) else ""
            print(f"{r['it']:<6} | {r['auc']*100:<10.2f} | {r['f1']*100:<8.2f} | {r['acc']*100:<8.2f} | {r['recall']*100:<11.2f} | {r['specificity']*100:<11.2f} | {r['loss']:<7.4f}{is_peak}")
    print("="*85)

    best_r = max(records, key=lambda x: x['auc'])
    final_r = records[-1]

    print("\n------------------------------------------------------------")
    print("PEAK PERFORMANCE SUMMARY (Best AUROC Checkpoint):")
    print(f"  - Peak Iteration   : {best_r['it']} / 4000")
    print(f"  - AUROC            : {best_r['auc']*100:.4f}% ({best_r['auc']:.6f})")
    print(f"  - F1-Score         : {best_r['f1']*100:.4f}% ({best_r['f1']:.6f})")
    print(f"  - Accuracy         : {best_r['acc']*100:.4f}% ({best_r['acc']:.6f})")
    print(f"  - Sensitivity (TPR): {best_r['recall']*100:.4f}% ({best_r['recall']:.6f})")
    print(f"  - Specificity (TNR): {best_r['specificity']*100:.4f}% ({best_r['specificity']:.6f})")
    print(f"  - Train Loss       : {best_r['loss']:.4f}")
    print("------------------------------------------------------------")
    print("FINAL PERFORMANCE SUMMARY (4000 iters):")
    print(f"  - Final Iteration  : {final_r['it']}")
    print(f"  - Final AUROC      : {final_r['auc']*100:.4f}% ({final_r['auc']:.6f})")
    print(f"  - Final F1-Score   : {final_r['f1']*100:.4f}% ({final_r['f1']:.6f})")
    print(f"  - Final Accuracy   : {final_r['acc']*100:.4f}% ({final_r['acc']:.6f})")
    print(f"  - Final Sensitivity: {final_r['recall']*100:.4f}% ({final_r['recall']:.6f})")
    print(f"  - Final Specificity: {final_r['specificity']*100:.4f}% ({final_r['specificity']:.6f})")
    print("------------------------------------------------------------")
