import pandas as pd
import numpy as np
import os
import time
import torch
import sympy as sp
from sklearn.metrics import r2_score
import hydra
import warnings
import visymre_utils as vu

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIRST_PRINCIPLES_ROOT = os.path.join(SCRIPT_DIR, 'datasets_first', 'firstprinciples')

target_datasets = [
    'first_principles_bode', 'first_principles_hubble', 'first_principles_kepler',
    'first_principles_tully_fisher', 'first_principles_planck', 'first_principles_ideal_gas',
    'first_principles_leavitt', 'first_principles_newton', 'first_principles_rydberg',
    'first_principles_schechter', 'first_principles_absorption',
    'first_principles_supernovae_zr', 'first_principles_supernovae_zg'
]


def load_srbench_data(dataset_name):
    possible_paths = [
        os.path.join(FIRST_PRINCIPLES_ROOT, dataset_name, f"{dataset_name}.tsv.gz"),
        os.path.join(FIRST_PRINCIPLES_ROOT, dataset_name, f"{dataset_name}.tsv")
    ]
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not file_path:
        print(f"[Error] Data file not found: {dataset_name}")
        return None, None

    try:
        df = pd.read_csv(file_path, sep='\t', compression='gzip' if file_path.endswith('.gz') else None, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        data_values = df.values.astype(np.float32)
        if data_values.shape[0] == 0: return None, None
        return data_values[:, :-1], data_values[:, -1]
    except Exception as e:
        print(f"[Read Failed] {dataset_name}: {e}")
        return None, None


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path)

    total_start = time.time()

    for run_idx in range(5):  # Run 5 times
        for ds_name in target_datasets:
            print(f"\n{'=' * 60}\nProcessing dataset: {ds_name}")

            X_raw, y_raw = load_srbench_data(ds_name)
            if X_raw is None or len(X_raw) == 0: continue

            indices = np.random.permutation(len(y_raw))
            split_point = int(len(y_raw) * 0.75)
            train_idx, test_idx = indices[:split_point], indices[split_point:]

            X_train_full, y_train_full = X_raw[train_idx], y_raw[train_idx]
            X_test_full, y_test_full = X_raw[test_idx], y_raw[test_idx]

            best_record = None
            # Track best based on training performance
            best_r2_train = -np.inf

            start_time_ds = time.time()

            scaler_x_obj = vu.AutoMagnitudeScaler(centering=False)
            scaler_y_obj = vu.AutoMagnitudeScaler(centering=False)

            for iter_idx in range(20):  # Iterations

                n_samples = 200
                sub_indices = np.random.choice(len(X_train_full), size=n_samples, replace=len(X_train_full) < n_samples)
                X_curr, y_curr = X_train_full[sub_indices], y_train_full[sub_indices]

                apply_scaling = (iter_idx >= 10)

                if apply_scaling:
                    scaler_x_obj.fit(X_train_full, y=y_train_full)
                    scaler_y_obj.fit(y_train_full)
                    X_input = scaler_x_obj.transform(X_curr)
                    y_input = scaler_y_obj.transform(y_curr)
                    scale_tag = "[Scaled]"
                else:
                    X_input, y_input = X_curr, y_curr
                    scale_tag = "[Raw   ]"

                # Input construction
                X_tensor = vu.pad_to_10_columns(torch.tensor(X_input, dtype=torch.float32))
                y_tensor = torch.tensor(y_input, dtype=torch.float32).reshape(-1, 1)

                try:
                    cfg.inference.beam_size = 150
                    output = fitfunc(X_tensor, y_tensor.squeeze(), cfg_params=cfg.inference, test_data=test_data)

                    pre_expr_sym = sp.sympify(output['best_bfgs_preds'][0])

                    # Restore expression
                    if apply_scaling:
                        expr_step1 = scaler_x_obj.restore_x_expression(pre_expr_sym)
                        final_expr_sym = scaler_y_obj.restore_y_expression(expr_step1)
                    else:
                        final_expr_sym = pre_expr_sym

                    final_expr_str = str(final_expr_sym)

                    # --- 1. Evaluate on Training Set (for Selection) ---
                    variables = vu.get_variable_names(final_expr_str)
                    func = None
                    if variables:
                        func = sp.lambdify(variables, final_expr_sym, modules="numpy")

                    if not variables:
                        y_pred_train = np.full(len(y_train_full),
                                               float(final_expr_sym) if final_expr_sym.is_Number else 0.0)
                    else:
                        X_eval_train = {}
                        for var in variables:
                            idx = int(var.split('_')[1]) - 1
                            X_eval_train[var] = X_train_full[:, idx] if idx < X_train_full.shape[1] else np.zeros(
                                len(y_train_full))
                        y_pred_train = func(**X_eval_train)

                    if isinstance(y_pred_train, (float, int)): y_pred_train = np.full(len(y_train_full), y_pred_train)
                    if np.iscomplexobj(y_pred_train): y_pred_train = y_pred_train.real
                    y_pred_train = np.nan_to_num(y_pred_train, nan=0.0)

                    r2_train = r2_score(y_train_full, y_pred_train)

                    print(f"  [Iter {iter_idx:02d}] {scale_tag} Train R2: {r2_train:.4f} | {final_expr_str}")

                    # --- 2. Update Best Record if Train R2 improves ---
                    if r2_train > best_r2_train:
                        best_r2_train = r2_train

                        # Calculate Test R2 (Reporting only)
                        if not variables:
                            y_pred_test = np.full(len(y_test_full),
                                                  float(final_expr_sym) if final_expr_sym.is_Number else 0.0)
                        else:
                            X_eval_test = {}
                            for var in variables:
                                idx = int(var.split('_')[1]) - 1
                                X_eval_test[var] = X_test_full[:, idx] if idx < X_test_full.shape[1] else np.zeros(
                                    len(y_test_full))
                            y_pred_test = func(**X_eval_test)

                        if isinstance(y_pred_test, (float, int)): y_pred_test = np.full(len(y_test_full), y_pred_test)
                        if np.iscomplexobj(y_pred_test): y_pred_test = y_pred_test.real
                        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0)

                        r2_test = r2_score(y_test_full, y_pred_test)

                        best_record = {
                            'dataset': ds_name, 'true_expr': "Unknown", 'pre_expr': final_expr_str,
                            'r2_test': r2_test, 'r2_train': r2_train,
                            'sr': -1, 'complexity': vu.calculate_tree_size(final_expr_str),
                            'time': time.time() - start_time_ds, 'beam': 50, 'scaled': apply_scaling,
                            'run': run_idx
                        }

                    # --- 3. Early Stopping based on Train R2 ---
                    if r2_train > 0.999:
                        break

                except Exception as e:
                    continue

            if best_record:
                print(f"Final {ds_name}: Test R2={best_record['r2_test']:.4f} | Expr={best_record['pre_expr']}")

                csv_file = 'First_Principles_test.csv'
                # Ensure we save the test R2 as the main 'r2' column for consistency with other tools, or keep distinct
                save_record = best_record.copy()
                save_record['r2'] = best_record['r2_test']  # Main R2 column is Test R2

                df_curr = pd.DataFrame([save_record])
                df_curr.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)
                print(f"  >>> Saved result to {csv_file}")

    print(f"\nCompleted. Total time: {(time.time() - total_start):.2f}s")


if __name__ == "__main__":
    main()
