import os
import time
import numpy as np
import pandas as pd
import torch
import sympy as sp
from sklearn.metrics import r2_score
import hydra
import re
import ast
import warnings

import visymre_utils as vu

warnings.filterwarnings("ignore")


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    # Setup
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path=metadata_path)

    benchmarks = pd.read_csv(r'C:\Users\lida\Desktop\visymre_10\scripts\low_benchmarks.csv').to_numpy()

    for line in benchmarks:
        raw_expr = line[2]
        expr_str = vu.replace_variables(str(raw_expr))

        expr_str = re.sub(r'\bx(\d+)\b', lambda m: f"x_{int(m.group(1)) + 1}", expr_str)
        variables = vu.get_variable_names(expr_str)

        print(f"Target: {expr_str}")

        # 1. Sampling
        original_range = ast.literal_eval(line[3])
        lam = vu.expr_to_func(sp.sympify(expr_str), variables)

        # Train / Test Generation
        points_train, y_train_raw = vu.sample_points(lam, len(variables), range_=original_range, target_noise=0.0)
        X_train_raw = points_train[:, :-1]

        points_test, y_test_raw = vu.sample_points(lam, len(variables), range_=original_range, target_noise=0.0)
        X_test_raw = points_test[:, :-1]

        # Initialize best, using Train R2 as selection criteria
        best = {"r2_train": -np.inf, "r2_test": -np.inf, "expr": ""}

        for beam in [3, 10, 20, 30, 50, 100]:
            cfg.inference.beam_size = beam

            # Scaler Selection
            if beam == 100:
                scaler_x = vu.AutoMagnitudeScaler(centering=False).fit(X_train_raw, y_train_raw)
                scaler_y = vu.AutoMagnitudeScaler(centering=False).fit(y_train_raw)
            else:
                scaler_x = vu.IdentityScaler()
                scaler_y = vu.IdentityScaler()

            # Transform
            X_t = vu.pad_to_10_columns(torch.tensor(scaler_x.transform(X_train_raw)).float())
            y_t = torch.tensor(scaler_y.transform(y_train_raw)).float().view(-1)

            try:
                out = fitfunc(X_t, y_t, cfg_params=cfg.inference, test_data=test_data)

                # Restore
                scaled_expr = sp.sympify(out['best_bfgs_preds'][0])
                restored_expr = scaler_y.restore_y_expression(scaler_x.restore_x_expression(scaled_expr))
                final_str = str(sp.simplify(restored_expr))

                pred_vars = vu.get_variable_names(final_str)

                # 1. Calculate Training R2
                if not pred_vars:
                    y_pred_train = np.full_like(y_train_raw, float(sp.sympify(final_str)))
                else:
                    X_dict_train = {v: X_train_raw[:, int(v.split('_')[1]) - 1] for v in pred_vars if
                                    int(v.split('_')[1]) - 1 < X_train_raw.shape[1]}
                    f_pred = sp.lambdify(pred_vars, sp.sympify(final_str), modules="numpy")
                    y_pred_train = f_pred(**X_dict_train)

                y_pred_train = np.nan_to_num(y_pred_train.real if np.iscomplexobj(y_pred_train) else y_pred_train,
                                             nan=0.0)
                r2_train = r2_score(y_train_raw, y_pred_train)

                # 2. Update Best only if Train R2 improves
                is_best = False
                current_test_r2 = -np.inf

                if r2_train > best["r2_train"]:
                    is_best = True
                    # Calculate Test R2 (for logging only, not for selection)
                    if not pred_vars:
                        y_pred_test = np.full_like(y_test_raw, float(sp.sympify(final_str)))
                    else:
                        X_dict_test = {v: X_test_raw[:, int(v.split('_')[1]) - 1] for v in pred_vars if
                                       int(v.split('_')[1]) - 1 < X_test_raw.shape[1]}
                        f_pred_test = sp.lambdify(pred_vars, sp.sympify(final_str), modules="numpy")
                        y_pred_test = f_pred_test(**X_dict_test)

                    y_pred_test = np.nan_to_num(y_pred_test.real if np.iscomplexobj(y_pred_test) else y_pred_test,
                                                nan=0.0)
                    current_test_r2 = r2_score(y_test_raw, y_pred_test)

                    best = {"r2_train": r2_train, "r2_test": current_test_r2, "expr": final_str}

                print(
                    f"Beam {beam}: Train R2={r2_train:.4f} | Test R2={current_test_r2 if is_best else '(skip)'} | {final_str}")

                # 3. Early Stopping (based on Train R2)
                if r2_train >= 0.999:
                    print("Early stopping based on Train R2.")
                    break

            except Exception as e:
                print(e)

        print(
            f"Final Selection -> Train R2: {best['r2_train']:.4f}, Test R2: {best['r2_test']:.4f}, Expr: {best['expr']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
