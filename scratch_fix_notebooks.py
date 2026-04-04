import json
import os

def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if line.startswith("    %cd "):
                    # Extract the path from '%cd path'
                    dir_path = line.replace("    %cd ", "").strip()
                    new_source.append(f"    os.chdir('{dir_path}')\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_notebook('notebooks/02_baseline_unet.ipynb')
fix_notebook('notebooks/03_multimodal_unet.ipynb')
print("Notebooks fixed successfully.")
