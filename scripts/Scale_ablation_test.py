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

# === 引入公共库 ===
import visymre_utils as vu

warnings.filterwarnings("ignore")


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    SCALER_TYPE = 'auto'
    print(f"=== Scaler Type: {SCALER_TYPE} ===")
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg,
                                                   metadata_path=metadata_path)

    benchmarks = pd.read_csv(r'C:\Users\lida\Desktop\visymre_10\scripts\low_benchmarks_scale.csv').to_numpy()
    results = []

    for line in benchmarks:
        raw_expr = line[2]
        expr_str = vu.replace_variables(str(raw_expr))
        expr_str = re.sub(r'\bx(\d+)\b', lambda m: f"x_{int(m.group(1)) + 1}", expr_str)
        variables = vu.get_variable_names(expr_str)
        print(f"Target: {expr_str}")

        for run_idx in range(1):
            try:
                # 采样
                range_ = [r * 1 for r in ast.literal_eval(line[3])]
                lam = vu.expr_to_func(sp.sympify(expr_str), variables)

                points_train, y_train_raw = vu.sample_points(lam, len(variables), range_=range_, target_noise=0.0)
                X_train_raw = points_train[:, :-1]

                points_test, y_test_raw = vu.sample_points(lam, len(variables), range_=range_, target_noise=0.0)
                X_test_raw = points_test[:, :-1]

                # 初始化 Scaler
                if SCALER_TYPE == 'zscore':
                    scaler_x = vu.ZScoreScaler().fit(X_train_raw)
                    scaler_y = vu.ZScoreScaler().fit(y_train_raw)
                elif SCALER_TYPE == 'minmax':
                    scaler_x = vu.MinMaxScaler().fit(X_train_raw)
                    scaler_y = vu.MinMaxScaler().fit(y_train_raw)
                else:
                    scaler_x = vu.AutoMagnitudeScaler(centering=False).fit(X_train_raw, y=y_train_raw)
                    scaler_y = vu.AutoMagnitudeScaler(centering=False).fit(y_train_raw)

                # 转换数据
                X_t = vu.pad_to_10_columns(torch.tensor(scaler_x.transform(X_train_raw)).float())
                y_t = torch.tensor(scaler_y.transform(y_train_raw)).float().view(-1)

                best_info = {"expr": "", "r2": -np.inf, "sr": 0, "cpx": 0, "time": 0}
                t_start = time.time()

                for beam in [3, 10, 20, 30, 50, 100]:
                    cfg.inference.beam_size = beam
                    try:
                        out = fitfunc(X_t, y_t, cfg_params=cfg.inference, test_data=test_data)

                        # 还原
                        scaled_expr = sp.sympify(out['best_bfgs_preds'][0])
                        final_expr_obj = scaler_y.restore_y_expression(scaler_x.restore_x_expression(scaled_expr))
                        final_expr_str = str(sp.simplify(final_expr_obj))

                        # 评估
                        pred_vars = vu.get_variable_names(final_expr_str)
                        if not pred_vars:
                            y_pred = np.full_like(y_test_raw,
                                                  float(final_expr_obj) if final_expr_obj.is_Number else 0.0)
                        else:
                            X_dict = {v: X_test_raw[:, int(v.split('_')[1]) - 1] for v in pred_vars if
                                      int(v.split('_')[1]) - 1 < X_test_raw.shape[1]}
                            func_pred = sp.lambdify(pred_vars, sp.sympify(final_expr_str), modules="numpy")
                            y_pred = func_pred(**X_dict)

                        y_pred = np.nan_to_num(y_pred.real if np.iscomplexobj(y_pred) else y_pred, nan=0.0)
                        r2 = r2_score(y_test_raw, y_pred)

                        print(f"Beam {beam}: R2={r2:.5f} | Expr={final_expr_str}")

                        if r2 > best_info["r2"]:
                            best_info.update({
                                "r2": r2, "expr": final_expr_str,
                                "sr": vu.symbol_equivalence_single(expr_str, final_expr_str, variables),
                                "cpx": vu.calculate_tree_size(final_expr_str),
                                "time": time.time() - t_start
                            })
                        if r2 >= 0.999: break

                    except Exception as e:
                        print(f"Error Beam {beam}: {e}")

                results.append([line[0], expr_str, best_info["expr"], best_info["r2"], best_info["sr"],
                                best_info["time"], best_info["cpx"], 0.0, range_])
                pd.DataFrame(results,
                             columns=['expr', 'true_expr', 'predict_expr', 'r2', 'sr', 'time', 'complexity', 'noise',
                                      'range']) \
                    .to_csv(f'visymre_result_{SCALER_TYPE}.csv', index=False)

            except Exception as e:
                print(f"Run Error: {e}")


if __name__ == "__main__":
    main()