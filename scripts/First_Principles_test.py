import pandas as pd
import numpy as np
import os
import torch
import sympy as sp
from sklearn.feature_selection import SelectKBest, r_regression, f_regression
from sklearn.metrics import r2_score
import hydra
import warnings
import visymre_utils as vu

warnings.filterwarnings("ignore")

# Config
TARGET_DATASETS = [
    "1028_SWD", "1089_USCrime", "1193_BNG_lowbwt", "1199_BNG_echoMonths",
    "192_vineyard", "210_cloud", "522_pm10", "557_analcatdata_apneal",
    "579_fri_c0_250_5", "606_fri_c2_1000_10", "650_fri_c0_500_50", "678_visualizing_environmental"
]
ITERATIONS = 30
SCALE_START_ITER = 15
SELECTED_K = 3
N_SAMPLES_PER_BAG = 200
BLACKBOX_DATA_DIR = "./srbench_blackbox_datasets"


def get_top_k_features(X, y, k=3):
    if y.ndim == 2: y = y[:, 0]
    if X.shape[1] <= k: return list(range(X.shape[1]))
    try:
        score_func = r_regression
    except NameError:
        score_func = f_regression
    selector = SelectKBest(score_func, k=k).fit(X, y)
    return list(np.argsort(-np.abs(selector.scores_))[:k])


def load_data(dataset_name):
    path = os.path.join(BLACKBOX_DATA_DIR, f"{dataset_name}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path).apply(pd.to_numeric, errors='coerce').dropna()
            return df.iloc[:, :-1].values, df.iloc[:, -1].values
        except Exception:
            return None, None
    return None, None


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    # 1. Initialize Model
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path)

    print(f"\n{'=' * 30} SRBench Bagging Evaluation {'=' * 30}")

    for _ in range(5):
        for ds_name in TARGET_DATASETS:
            print(f"\n>>> Dataset: {ds_name}")
            X_raw, y_raw = load_data(ds_name)
            if X_raw is None: continue

            # Feature Selection & Split
            top_indices = get_top_k_features(X_raw, y_raw, k=SELECTED_K)
            X_selected = X_raw[:, top_indices]

            idx = np.random.RandomState(42).permutation(len(y_raw))
            split = int(len(y_raw) * 0.75)
            X_train, y_train = X_selected[idx[:split]], y_raw[idx[:split]]
            X_test, y_test = X_selected[idx[split:]], y_raw[idx[split:]]

            # Pre-init Scalers
            scaler_x = vu.AutoMagnitudeScaler(centering=False)
            scaler_y = vu.AutoMagnitudeScaler(centering=False)

            best_r2_train = -np.inf
            best_record = None

            for iter_idx in range(ITERATIONS):
                apply_scaling = (iter_idx >= SCALE_START_ITER)

                # Bagging
                bag_idx = np.random.RandomState(iter_idx).choice(len(X_train),
                                                                 size=min(N_SAMPLES_PER_BAG, len(X_train)),
                                                                 replace=True)
                X_curr, y_curr = X_train[bag_idx], y_train[bag_idx]

                # Scaling
                scaler_x.fit(X_curr, y=y_curr)
                scaler_y.fit(y_curr)

                if apply_scaling:
                    X_in = scaler_x.transform(X_curr)
                    y_in = scaler_y.transform(y_curr)
                else:
                    X_in, y_in = X_curr, y_curr

                # Tensor Prep
                X_tensor = vu.pad_to_10_columns(torch.tensor(X_in, dtype=torch.float32))
                y_tensor = torch.tensor(y_in, dtype=torch.float32).reshape(-1, 1)

                try:
                    cfg.inference.beam_size = 150
                    output = fitfunc(X_tensor, y_tensor.squeeze(), cfg_params=cfg.inference, test_data=test_data)

                    pre_expr_sym = sp.sympify(output['best_bfgs_preds'][0])

                    # Restore Expression
                    if apply_scaling:
                        expr_s1 = scaler_x.restore_x_expression(pre_expr_sym)
                        final_expr_sym = scaler_y.restore_y_expression(expr_s1)
                    else:
                        final_expr_sym = pre_expr_sym

                    final_expr_str = str(final_expr_sym)

                    # Evaluation
                    variables = vu.get_variable_names(final_expr_str)
                    func = None
                    if variables:
                        func = sp.lambdify(variables, final_expr_sym, modules="numpy")

                    # 1. Evaluate on Train (Selection)
                    if not variables:
                        y_pred_train = np.full(len(y_train), float(final_expr_sym) if final_expr_sym.is_Number else 0.0)
                    else:
                        X_eval_train = {}
                        for var in variables:
                            idx = int(var.split('_')[1]) - 1
                            X_eval_train[var] = X_train[:, idx] if idx < X_train.shape[1] else np.zeros(len(y_train))
                        y_pred_train = func(**X_eval_train)

                    if isinstance(y_pred_train, (float, int)): y_pred_train = np.full(len(y_train), y_pred_train)
                    if np.iscomplexobj(y_pred_train): y_pred_train = y_pred_train.real
                    y_pred_train = np.nan_to_num(y_pred_train, nan=0.0)

                    r2_train = r2_score(y_train, y_pred_train)

                    # Update Best
                    if r2_train > best_r2_train:
                        best_r2_train = r2_train

                        # 2. Evaluate on Test (Reporting)
                        if not variables:
                            y_pred_test = np.full(len(y_test),
                                                  float(final_expr_sym) if final_expr_sym.is_Number else 0.0)
                        else:
                            X_eval_test = {}
                            for var in variables:
                                idx = int(var.split('_')[1]) - 1
                                X_eval_test[var] = X_test[:, idx] if idx < X_test.shape[1] else np.zeros(len(y_test))
                            y_pred_test = func(**X_eval_test)

                        if isinstance(y_pred_test, (float, int)): y_pred_test = np.full(len(y_test), y_pred_test)
                        if np.iscomplexobj(y_pred_test): y_pred_test = y_pred_test.real
                        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0)

                        r2_test = r2_score(y_test, y_pred_test)

                        best_record = {
                            'dataset': ds_name, 'r2_test': r2_test, 'r2_train': r2_train,
                            'expression': final_expr_str,
                            'scaled': apply_scaling, 'iter': iter_idx,
                            'complexity': vu.calculate_tree_size(final_expr_str)
                        }
                        print(f"  Iter {iter_idx:02d} New Best Train R2: {r2_train:.4f} (Test R2: {r2_test:.4f})")

                    # Early Stopping based on Train
                    if best_r2_train > 0.999: break

                except Exception:
                    pass

            if best_record:
                print(f"Final {ds_name}: Test R2={best_record['r2_test']:.4f} | Expr={best_record['expression']}")
                df = pd.DataFrame([best_record])
                df.to_csv('visymre_result_black_box.csv', mode='a',
                          header=not os.path.isfile('visymre_result_black_box.csv'), index=False)


if __name__ == "__main__":
    main()
