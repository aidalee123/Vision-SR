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
    results = []

    for line in benchmarks:
        raw_expr = line[2]
        expr_str = vu.replace_variables(str(raw_expr))
        # 修正变量编号 x0 -> x_1
        expr_str = re.sub(r'\bx(\d+)\b', lambda m: f"x_{int(m.group(1)) + 1}", expr_str)
        variables = vu.get_variable_names(expr_str)

        print(f"Target: {expr_str}")

        # 1. 采样
        original_range = ast.literal_eval(line[3])
        lam = vu.expr_to_func(sp.sympify(expr_str), variables)

        # Train / Test Generation (使用 utils)
        points_train, y_train_raw = vu.sample_points(lam, len(variables), range_=original_range, target_noise=0.0)
        X_train_raw = points_train[:, :-1]

        points_test, y_test_raw = vu.sample_points(lam, len(variables), range_=original_range, target_noise=0.0)
        X_test_raw = points_test[:, :-1]

        best = {"r2": -np.inf, "expr": ""}

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

                # Evaluate R2 on Raw
                pred_vars = vu.get_variable_names(final_str)
                if not pred_vars:
                    y_pred = np.full_like(y_test_raw, float(sp.sympify(final_str)))
                else:
                    X_dict = {v: X_test_raw[:, int(v.split('_')[1]) - 1] for v in pred_vars if
                              int(v.split('_')[1]) - 1 < X_test_raw.shape[1]}
                    f_pred = sp.lambdify(pred_vars, sp.sympify(final_str), modules="numpy")
                    y_pred = f_pred(**X_dict)

                y_pred = np.nan_to_num(y_pred.real if np.iscomplexobj(y_pred) else y_pred, nan=0.0)
                r2 = r2_score(y_test_raw, y_pred)
                # sr = vu.symbol_equivalence_single(str(sp.sympify(formula_str)),
                #                                   str(vu.coefficient_regularization(str(pre_expr))), variables)

                print(f"Beam {beam}: R2={r2:.4f} | {final_str}")

                if r2 > best["r2"]:
                    best = {"r2": r2, "expr": final_str}
                if r2 >= 0.999: break

            except Exception as e:
                print(e)

        results.append([line[0], expr_str, best["expr"], best["r2"]])
        pd.DataFrame(results).to_csv('visymre_result_mixed.csv', index=False)


if __name__ == "__main__":
    main()