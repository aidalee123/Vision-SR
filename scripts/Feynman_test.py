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

FEYNMAN_DATA_DIR = './Feynman_with_units'
FEYNMAN_LABELS_FILE = './FeynmanEquations.xlsx'


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path=metadata_path)

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
            # Initialize lists to store history (optional) and best tracker
            r2_train_history, r2_test_history, pre_expr_history = [], [], []
            best = {"r2_train": -np.inf, "r2_test": -np.inf, "expr": "-", "time": 0}
            
            start_time = time.time()

            for beam in [30]:
                cfg.inference.beam_size = beam

                # Split
                idx = np.random.permutation(len(data))
                split = int(len(data) * 0.75)
                train_set, test_set = data[idx[:split]], data[idx[split:]]

                # Add Noise (Target)
                scale = target_noise * np.sqrt(np.mean(np.square(train_set[:, -1])))
                noise = np.random.normal(0, scale, train_set[:, -1].shape)
                train_set[:, -1] += noise
                
                # Prepare Full Train/Test Tensors for Evaluation
                X_train_full_t = vu.pad_to_10_columns(torch.tensor(train_set[:, :-1]))
                y_train_full = train_set[:, -1]
                X_test_full_t = vu.pad_to_10_columns(torch.tensor(test_set[:, :-1]))
                y_test_full = test_set[:, -1]

                for _ in range(8):  # Iterations
                    cfg.inference.beam_size = min(30, (_ + 1) * 10)

                    # Subsample for Training
                    sub_idx = np.random.choice(train_set.shape[0], size=300, replace=False)
                    X_train_batch = train_set[sub_idx, :-1]
                    y_train_batch = train_set[sub_idx, -1:]

                    X_t = vu.pad_to_10_columns(torch.tensor(X_train_batch))
                    y_t = torch.tensor(y_train_batch)

                    try:
                        output = fitfunc(X_t, y_t.squeeze(), cfg_params=cfg.inference, test_data=test_data)
                        pre_expr = sp.sympify(output['best_bfgs_preds'][0])
                        vars_ = vu.get_variable_names(str(pre_expr))
                        
                        func = sp.lambdify(vars_, pre_expr, modules="numpy")

                        # 1. Evaluate on Full Training Set (Selection Metric)
                        X_train_dict = {v: X_train_full_t[:, i].cpu() for i, v in enumerate(vars_)}
                        y_pre_train = func(**X_train_dict)
                        # Handle complex/NaN
                        y_pre_train = np.nan_to_num(y_pre_train.real if np.iscomplexobj(y_pre_train) else y_pre_train, nan=0.0)
                        
                        r2_train = r2_score(y_train_full, y_pre_train)

                        # 2. Update Best & Evaluate on Test Set (Reporting Metric)
                        if r2_train > best["r2_train"]:
                            X_test_dict = {v: X_test_full_t[:, i].cpu() for i, v in enumerate(vars_)}
                            y_pre_test = func(**X_test_dict)
                            y_pre_test = np.nan_to_num(y_pre_test.real if np.iscomplexobj(y_pre_test) else y_pre_test, nan=0.0)
                            
                            r2_test = r2_score(y_test_full, y_pre_test)
                            
                            best.update({
                                "r2_train": r2_train,
                                "r2_test": r2_test,
                                "expr": str(pre_expr)
                            })
                        
                        r2_train_history.append(r2_train)
                        pre_expr_history.append(str(pre_expr))

                        # Early Stopping based on Train R2
                        if best["r2_train"] > 0.999: break

                    except Exception as e:
                        r2_train_history.append(0)
                        pre_expr_history.append("-")

                if best["r2_train"] >= 0.999: break

            elapsed = time.time() - start_time

            print(f"Final {filename}: Test R2={best['r2_test']:.4f} | Expr={best['expr']}")

if __name__ == "__main__":
    main()
