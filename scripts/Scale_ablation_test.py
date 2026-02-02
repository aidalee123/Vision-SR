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
    SCALER_TYPE = 'auto'
    print(f"=== Scaler Type: {SCALER_TYPE} ===")
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path=metadata_path)

    benchmarks = pd.read_csv(r'.\low_benchmarks_scale.csv').to_numpy()

    for line in benchmarks:
        raw_expr = line[2]
        expr_str = vu.replace_variables(str(raw_expr))
        expr_str = re.sub(r'\bx(\d+)\b', lambda m: f"x_{int(m.group(1)) + 1}", expr_str)
        variables = vu.get_variable_names(expr_str)
        print(f"Target: {expr_str}")

        for run_idx in range(1):
            try:
                # Sampling
                range_ = [r * 1 for r in ast.literal_eval(line[3])]
                lam = vu.expr_to_func(sp.sympify(expr_str), variables)

                points_train, y_train_raw = vu.sample_points(lam, len(variables), range_=range_, target_noise=0.0)
                X_train_raw = points_train[:, :-1]

                points_test, y_test_raw = vu.sample_points(lam, len(variables), range_=range_, target_noise=0.0)
                X_test_raw = points_test[:, :-1]

                if SCALER_TYPE == 'zscore':
                    scaler_x = vu.ZScoreScaler().fit(X_train_raw)
                    scaler_y = vu.ZScoreScaler().fit(y_train_raw)
                elif SCALER_TYPE == 'minmax':
                    scaler_x = vu.MinMaxScaler().fit(X_train_raw)
                    scaler_y = vu.MinMaxScaler().fit(y_train_raw)
                else:
                    scaler_x = vu.AutoMagnitudeScaler(centering=False).fit(X_train_raw, y=y_train_raw)
                    scaler_y = vu.AutoMagnitudeScaler(centering=False).fit(y_train_raw)

                X_t = vu.pad_to_10_columns(torch.tensor(scaler_x.transform(X_train_raw)).float())
                y_t = torch.tensor(scaler_y.transform(y_train_raw)).float().view(-1)

                best_info = {"expr": "", "r2_train": -np.inf, "r2_test": -np.inf, "sr": 0, "cpx": 0, "time": 0}
                t_start = time.time()

                for beam in [3, 10, 20, 30, 50, 100]:
                    cfg.inference.beam_size = beam
                    try:
                        out = fitfunc(X_t, y_t, cfg_params=cfg.inference, test_data=test_data)

                        # Restore
                        scaled_expr = sp.sympify(out['best_bfgs_preds'][0])
                        final_expr_obj = scaler_y.restore_y_expression(scaler_x.restore_x_expression(scaled_expr))
                        final_expr_str = str(sp.simplify(final_expr_obj))

                        # Evaluate on Train
                        pred_vars = vu.get_variable_names(final_expr_str)
                        func_pred = None
                        if pred_vars:
                            func_pred = sp.lambdify(pred_vars, sp.sympify(final_expr_str), modules="numpy")

                        if not pred_vars:
                            y_pred_train = np.full_like(y_train_raw,
                                                        float(final_expr_obj) if final_expr_obj.is_Number else 0.0)
                        else:
                            X_dict_train = {v: X_train_raw[:, int(v.split('_')[1]) - 1] for v in pred_vars if
                                            int(v.split('_')[1]) - 1 < X_train_raw.shape[1]}
                            y_pred_train = func_pred(**X_dict_train)

                        y_pred_train = np.nan_to_num(
                            y_pred_train.real if np.iscomplexobj(y_pred_train) else y_pred_train, nan=0.0)
                        r2_train = r2_score(y_train_raw, y_pred_train)

                        # Update best based on Train R2
                        if r2_train > best_info["r2_train"]:
                            # Evaluate on Test only if model is selected
                            if not pred_vars:
                                y_pred_test = np.full_like(y_test_raw,
                                                           float(final_expr_obj) if final_expr_obj.is_Number else 0.0)
                            else:
                                X_dict_test = {v: X_test_raw[:, int(v.split('_')[1]) - 1] for v in pred_vars if
                                               int(v.split('_')[1]) - 1 < X_test_raw.shape[1]}
                                y_pred_test = func_pred(**X_dict_test)

                            y_pred_test = np.nan_to_num(
                                y_pred_test.real if np.iscomplexobj(y_pred_test) else y_pred_test, nan=0.0)
                            r2_test = r2_score(y_test_raw, y_pred_test)

                            best_info.update({
                                "r2_train": r2_train,
                                "r2_test": r2_test,
                                "expr": final_expr_str,
                                "sr": vu.symbol_equivalence_single(expr_str, final_expr_str, variables),
                                "cpx": vu.calculate_tree_size(final_expr_str),
                                "time": time.time() - t_start
                            })

                        print(f"Beam {beam}: Train R2={r2_train:.5f} | Expr={final_expr_str}")

                        if r2_train >= 0.999:
                            break

                    except Exception as e:
                        print(f"Error Beam {beam}: {e}")

                print(f"Final {line[0]}: Test R2={best_info['r2_test']:.4f} | Expr={best_info['expr']}")

            except Exception as e:
                print(f"Run Error: {e}")


if __name__ == "__main__":
    main()
