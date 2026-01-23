import os
import numpy as np
import time
import pandas as pd
import torch
import sympy as sp
import hydra
from sklearn.metrics import r2_score
from functools import partial
import warnings

import visymre_utils as vu

warnings.filterwarnings("ignore")

FEYNMAN_DATA_DIR = '/home/lida/Desktop/visymre_no10v/scripts/Feynman_with_units'
FEYNMAN_LABELS_FILE = '/home/lida/Desktop/visymre_no10v/scripts/FeynmanEquations.xlsx'


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg,
                                                   metadata_path = metadata_path)

    files = [f for f in os.listdir(FEYNMAN_DATA_DIR) if os.path.isfile(os.path.join(FEYNMAN_DATA_DIR, f))]
    df_labels = pd.read_excel(FEYNMAN_LABELS_FILE)
    results = []

    for i in range(3):  # Run 3 times
        for filename in files:
            print(f"Processing: {filename}")
            try:
                formula_str = df_labels.loc[df_labels['Filename'] == filename, 'replaced_formula'].values[0]
            except:
                continue

            data = np.loadtxt(os.path.join(FEYNMAN_DATA_DIR, filename))

            target_noise = 0.01
            r2_list, solut, pre_expr_list = [], [], []
            start_time = time.time()

            for beam in [30]:
                cfg.inference.beam_size = beam

                # Split
                idx = np.random.permutation(len(data))
                split = int(len(data) * 0.75)
                train_set, test_set = data[idx[:split]], data[idx[split:]]

                # Add Noise
                scale = target_noise * np.sqrt(np.mean(np.square(train_set[:, -1])))
                noise = np.random.normal(0, scale, train_set[:, -1].shape)
                train_set[:, -1] += noise

                for _ in range(8):  # Iterations
                    cfg.inference.beam_size = min(30, (_ + 1) * 10)

                    sub_idx = np.random.choice(train_set.shape[0], size=300, replace=False)
                    X_train = train_set[sub_idx, :-1]
                    y_train = train_set[sub_idx, -1:]

                    X_t = vu.pad_to_10_columns(torch.tensor(X_train))
                    y_t = torch.tensor(y_train)

                    X_test_t = vu.pad_to_10_columns(torch.tensor(test_set[:, :-1]))

                    try:
                        output = fitfunc(X_t, y_t.squeeze(), cfg_params=cfg.inference, test_data=test_data)
                        pre_expr = sp.sympify(output['best_bfgs_preds'][0])

                        # Feynman特有的 Refinement 步骤 (可选调用 utils 中的优化器)
                        # refined_expr = vu.optimize_expression_constants(pre_expr, ...)
                        # 此处保持原逻辑简化版：

                        vars_ = vu.get_variable_names(str(pre_expr))
                        X_test_dict = {v: X_test_t[:, i].cpu() for i, v in enumerate(vars_)}

                        func = sp.lambdify(vars_, pre_expr, modules="numpy")
                        y_pre = func(**X_test_dict)

                        # SR Check (need utils import for equivalence if needed)
                        # sr = ...

                        r2 = r2_score(test_set[:, -1], y_pre.real) if not np.iscomplexobj(y_pre) else 0

                        r2_list.append(r2)
                        pre_expr_list.append(str(pre_expr))

                        if max(r2_list) > 0.999: break
                    except:
                        r2_list.append(0)
                        pre_expr_list.append("-")

                if max(r2_list) >= 0.999: break

            elapsed = time.time() - start_time
            best_idx = np.argmax(r2_list)

            results.append([
                filename, formula_str, pre_expr_list[best_idx],
                r2_list[best_idx], elapsed, target_noise, i
            ])

            pd.DataFrame(results).to_csv('visymre_feim_result.csv', index=False)


if __name__ == "__main__":
    main()