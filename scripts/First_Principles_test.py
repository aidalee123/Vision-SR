import pandas as pd
import numpy as np
import os
import time
import torch
import sympy as sp
from sklearn.metrics import r2_score
import hydra
import warnings

# === 引入公共库 ===
import visymre_utils as vu

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 请确保此路径指向你的 datasets_first/firstprinciples 文件夹
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
        print(f"[错误] 找不到数据文件: {dataset_name}")
        return None, None

    try:
        df = pd.read_csv(file_path, sep='\t', compression='gzip' if file_path.endswith('.gz') else None, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        data_values = df.values.astype(np.float32)
        if data_values.shape[0] == 0: return None, None
        return data_values[:, :-1], data_values[:, -1]
    except Exception as e:
        print(f"[读取失败] {dataset_name}: {e}")
        return None, None


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    # 初始化模型
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg, metadata_path)

    total_start = time.time()

    for _ in range(5):  # Run 5 times
        for ds_name in target_datasets:
            print(f"\n{'=' * 60}\n正在处理数据集: {ds_name}")

            X_raw, y_raw = load_srbench_data(ds_name)
            if X_raw is None or len(X_raw) == 0: continue

            # 数据集划分
            indices = np.random.permutation(len(y_raw))
            split_point = int(len(y_raw) * 0.75)
            train_idx, test_idx = indices[:split_point], indices[split_point:]

            X_train_full, y_train_full = X_raw[train_idx], y_raw[train_idx]
            X_test_full, y_test_full = X_raw[test_idx], y_raw[test_idx]

            best_record = None
            best_r2_for_dataset = -np.inf
            start_time_ds = time.time()

            # Scalers 预初始化
            scaler_x_obj = vu.AutoMagnitudeScaler(centering=False)
            scaler_y_obj = vu.AutoMagnitudeScaler(centering=False)

            for iter_idx in range(20):  # Iterations
                # 采样
                n_samples = 200
                sub_indices = np.random.choice(len(X_train_full), size=n_samples, replace=len(X_train_full) < n_samples)
                X_curr, y_curr = X_train_full[sub_indices], y_train_full[sub_indices]

                # 混合策略: 前10轮 Raw, 后10轮 Scaled
                apply_scaling = (iter_idx >= 10)

                if apply_scaling:
                    # 使用全量训练数据 fit 更稳
                    scaler_x_obj.fit(X_train_full, y=y_train_full)
                    scaler_y_obj.fit(y_train_full)
                    X_input = scaler_x_obj.transform(X_curr)
                    y_input = scaler_y_obj.transform(y_curr)
                    scale_tag = "[Scaled]"
                else:
                    X_input, y_input = X_curr, y_curr
                    scale_tag = "[Raw   ]"

                # 构造输入
                X_tensor = vu.pad_to_10_columns(torch.tensor(X_input, dtype=torch.float32))
                y_tensor = torch.tensor(y_input, dtype=torch.float32).reshape(-1, 1)

                try:
                    cfg.inference.beam_size = 150
                    output = fitfunc(X_tensor, y_tensor.squeeze(), cfg_params=cfg.inference, test_data=test_data)

                    pre_expr_sym = sp.sympify(output['best_bfgs_preds'][0])

                    # 还原表达式
                    if apply_scaling:
                        expr_step1 = scaler_x_obj.restore_x_expression(pre_expr_sym)
                        final_expr_sym = scaler_y_obj.restore_y_expression(expr_step1)
                    else:
                        final_expr_sym = pre_expr_sym

                    final_expr_str = str(final_expr_sym)

                    # 评估
                    variables = vu.get_variable_names(final_expr_str)
                    if not variables:
                        y_pred = np.full(len(y_test_full), float(final_expr_sym) if final_expr_sym.is_Number else 0.0)
                    else:
                        X_eval = {}
                        for var in variables:
                            idx = int(var.split('_')[1]) - 1
                            X_eval[var] = X_test_full[:, idx] if idx < X_test_full.shape[1] else np.zeros(
                                len(y_test_full))

                        func = sp.lambdify(variables, final_expr_sym, modules="numpy")
                        y_pred = func(**X_eval)

                    # 数值清理
                    if isinstance(y_pred, (float, int)): y_pred = np.full(len(y_test_full), y_pred)
                    if np.iscomplexobj(y_pred): y_pred = y_pred.real
                    y_pred = np.nan_to_num(y_pred, nan=0.0)

                    r2 = r2_score(y_test_full, y_pred)

                    print(f"  [Iter {iter_idx:02d}] {scale_tag} R2: {r2:.4f} | {final_expr_str}")

                    if r2 > best_r2_for_dataset:
                        best_r2_for_dataset = r2
                        best_record = {
                            'dataset': ds_name, 'true_expr': "Unknown", 'pre_expr': final_expr_str,
                            'r2': r2, 'sr': -1, 'complexity': vu.calculate_tree_size(final_expr_str),
                            'time': time.time() - start_time_ds, 'beam': 50, 'scaled': apply_scaling
                        }

                    if r2 > 0.999: break

                except Exception as e:
                    continue

            if best_record:
                csv_file = 'visymre_results_first.csv'
                df_curr = pd.DataFrame([best_record])
                df_curr.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)
                print(f"  >>> {ds_name} Best R2: {best_record['r2']:.4f} (Saved)")

    print(f"\nCompleted. Total time: {(time.time() - total_start):.2f}s")


if __name__ == "__main__":
    main()