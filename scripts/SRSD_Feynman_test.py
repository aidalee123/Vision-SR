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
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path=metadata_path)

    files = [f for f in os.listdir(TRAIN_SET_DIR) if os.path.isfile(os.path.join(TRAIN_SET_DIR, f))]

    for run in range(1):
        for filename in files:
            print(f"Processing: {filename}")
            try:
                train_set = np.loadtxt(os.path.join(TRAIN_SET_DIR, filename))
                test_set = np.loadtxt(os.path.join(TEST_SET_DIR, filename))
                expr_path = os.path.join(GROUND_TRUTH_DIR, filename.replace(".txt", ".pkl"))
                true_expr_str = transform_expression(str(pd.read_pickle(expr_path)))

                X_train_full = train_set[:, :-1]
                y_train_full = train_set[:, -1]
                X_test_np, y_test_np = test_set[:, :-1], test_set[:, -1]

                sx = vu.AutoMagnitudeScaler(centering=False).fit(train_set[:, :-1], y=train_set[:, -1])
                sy = vu.AutoMagnitudeScaler(centering=False).fit(train_set[:, -1])

                best = {"r2_train": -np.inf, "r2_test": -np.inf, "sr": 0, "cpx": 0, "expr": "", "time": 0}
                t0 = time.time()
                n_ = 0

                for beam in [30]:
                    cfg.inference.beam_size = beam
                    for _ in range(10):
                        idx = np.random.choice(train_set.shape[0], size=200, replace=False)
                        X_batch, y_batch = train_set[idx, :-1], train_set[idx, -1].astype(np.float32)

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

                            if _ >= n_:
                                pred_expr = sy.restore_y_expression(sx.restore_x_expression(pred_expr))

                            final_expr_str = str(pred_expr)
                            vars_ = vu.get_variable_names(final_expr_str)
                            f = sp.lambdify(vars_, sp.sympify(final_expr_str), modules=["numpy"])

                            args_train = [X_train_full[:, i] for i in range(len(vars_))]
                            if not args_train and not vars_:
                                y_pred_train = np.full_like(y_train_full, float(pred_expr))
                            else:
                                y_pred_train = np.asarray(f(*args_train)).reshape(-1)

                            y_pred_train = np.nan_to_num(y_pred_train.real if np.iscomplexobj(y_pred_train) else y_pred_train, nan=1e9)
                            r2_train = r2_score(y_train_full, y_pred_train)

                            if r2_train > best["r2_train"]:
                                args_test = [X_test_np[:, i] for i in range(len(vars_))]
                                if not args_test and not vars_:
                                    y_pred_test = np.full_like(y_test_np, float(pred_expr))
                                else:
                                    y_pred_test = np.asarray(f(*args_test)).reshape(-1)

                                y_pred_test = np.nan_to_num(y_pred_test.real if np.iscomplexobj(y_pred_test) else y_pred_test, nan=1e9)
                                r2_test = r2_score(y_test_np, y_pred_test)

                                best.update({"r2_train": r2_train, "r2_test": r2_test, "sr": 0, "expr": final_expr_str,
                                             "cpx": vu.calculate_tree_size(final_expr_str), "time": time.time() - t0})

                            if best["r2_train"] >= 0.999: break
                        except Exception:
                            pass

                    if best["r2_train"] >= 0.999: break

                print(f"Final {filename}: Test R2={best['r2_test']:.4f} | Expr={best['expr']}")


            except Exception as e:
                print(f"Error {filename}: {e}")


if __name__ == "__main__":
    main()
