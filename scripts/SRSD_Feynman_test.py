import os
import time
import numpy as np
import pandas as pd
import torch
import sympy as sp
from sklearn.metrics import r2_score
import hydra
import re
import warnings


# === 引入公共库 ===
import visymre_utils as vu

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_SET_DIR = "/home/lida/Desktop/visymre_no10v/scripts/srsd_benchmark-feynman/medium/train"
TEST_SET_DIR = "/home/lida/Desktop/visymre_no10v/scripts/srsd_benchmark-feynman/medium/test"
GROUND_TRUTH_DIR = "/home/lida/Desktop/visymre_no10v/scripts/srsd_benchmark-feynman/medium/true_expr"


def transform_expression(expression):
    return re.sub(r'\bx(\d+)\b', lambda m: f"x_{int(m.group(1)) + 1}", expression)


@hydra.main(config_name="config", version_base="1.1")
def main(cfg):
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg,
                                                   metadata_path =metadata_path)

    files = [f for f in os.listdir(TRAIN_SET_DIR) if os.path.isfile(os.path.join(TRAIN_SET_DIR, f))]
    results = []

    for run in range(1):
        for filename in files:
            print(f"Processing: {filename}")
            try:
                # Load Data
                train_set = np.loadtxt(os.path.join(TRAIN_SET_DIR, filename))
                test_set = np.loadtxt(os.path.join(TEST_SET_DIR, filename))
                expr_path = os.path.join(GROUND_TRUTH_DIR, filename.replace(".txt", ".pkl"))
                true_expr_str = transform_expression(str(pd.read_pickle(expr_path)))

                X_test_np, y_test_np = test_set[:, :-1], test_set[:, -1]

                # Initialize Scalers
                sx = vu.AutoMagnitudeScaler(centering=False).fit(train_set[:, :-1], y=train_set[:, -1])
                sy = vu.AutoMagnitudeScaler(centering=False).fit(train_set[:, -1])

                best = {"r2": -np.inf, "sr": 0, "cpx": 0, "expr": "", "time": 0}
                t0 = time.time()
                n_ = 0  # Switch scaling iteration threshold

                for beam in [30]:
                    cfg.inference.beam_size = beam
                    for _ in range(10):  # Iterations
                        idx = np.random.choice(train_set.shape[0], size=200, replace=False)
                        X_batch, y_batch = train_set[idx, :-1], train_set[idx, -1].astype(np.float32)

                        # Scaling Strategy
                        if _ >= n_:
                            X_in = sx.transform(X_batch)
                            y_in = sy.transform(y_batch).ravel()
                        else:
                            X_in, y_in = X_batch.astype(np.float32), y_batch.astype(np.float32)

                        X_t = vu.pad_to_10_columns(torch.tensor(X_in))
                        y_t = torch.tensor(y_in)

                        try:
                            out = fitfunc(X_t, y_t.squeeze(), cfg_params=cfg.inference, test_data=test_data)
                            pred_expr = sp.sympify(out["best_bfgs_preds"][0])

                            # Restore
                            if _ >= n_:
                                pred_expr = sy.restore_y_expression(sx.restore_x_expression(pred_expr))

                            final_expr_str = str(pred_expr)

                            # Eval
                            vars_ = vu.get_variable_names(final_expr_str)
                            f = sp.lambdify(vars_, sp.sympify(final_expr_str), modules=["numpy"])

                            args = [X_test_np[:, i] for i in range(len(vars_))]
                            if not args and not vars_:
                                y_pred = np.full_like(y_test_np, float(pred_expr))
                            else:
                                y_pred = np.asarray(f(*args)).reshape(-1)

                            y_pred = np.nan_to_num(y_pred.real if np.iscomplexobj(y_pred) else y_pred, nan=1e9)
                            r2 = r2_score(y_test_np, y_pred)

                            # Update Best
                            if r2 > best["r2"]:
                                # sr = compare_equation_from_str(final_expr_str, true_expr_str, normalizes=True,
                                #                                decrements_idx=False)
                                best.update({"r2": r2, "sr": sr, "expr": final_expr_str,
                                             "cpx": vu.calculate_tree_size(final_expr_str), "time": time.time() - t0})

                            if best["r2"] >= 0.999: break
                        except Exception:
                            pass

                    if best["r2"] >= 0.999: break

                results.append([filename, true_expr_str, best["expr"], best["r2"], best["sr"],
                                best["time"], best["cpx"], 0.0, run])

                pd.DataFrame(results,
                             columns=["filename", "true_expr", "pre_expr", "r2", "sr", "t", "Complexity", "noise",
                                      "Iteration"]) \
                    .to_csv("hard_norm_result.csv", index=False)

            except Exception as e:
                print(f"Error {filename}: {e}")


if __name__ == "__main__":
    main()