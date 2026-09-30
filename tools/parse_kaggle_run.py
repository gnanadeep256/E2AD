import json

with open('notebook8d611b5bf2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(len(nb['cells'])):
    cell = nb['cells'][idx]
    if cell['cell_type'] == 'code':
        print(f"\n{'='*70}")
        print(f"CELL {idx+1} [exec_count={cell.get('execution_count')}]")
        print(f"{'='*70}")
        # print first 2 lines of code
        code_lines = cell['source'][:2]
        print("Code preview:", "".join(code_lines).strip())
        print("-" * 50)
        outputs = cell.get('outputs', [])
        for out in outputs:
            if 'text' in out:
                txt = ''.join(out['text'])
                # If output is very long, print first few and last few lines
                lines = txt.splitlines()
                if len(lines) > 40:
                    print("\n".join(lines[:20]))
                    print(f"\n... [{len(lines)-35} lines omitted] ...\n")
                    print("\n".join(lines[-15:]))
                else:
                    print(txt)
            elif 'data' in out:
                print(out['data'].get('text/plain', ''))
