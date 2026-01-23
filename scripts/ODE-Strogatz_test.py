import os
import numpy as np
import time
import pandas as pd
import torch
import sympy as sp
from sklearn.metrics import r2_score
import hydra
import warnings

# === 引入公共库 ===
import visymre_utils as vu

warnings.filterwarnings("ignore")

ODE_DATA_DIR = '/home/lida/Desktop/visymre_no10v/scripts/ode-strogatz-master/ode-strogatz-master'
ODE_LABELS_FILE = '/home/lida/Desktop/visymre_no10v/scripts/ode-strogatz-master/ode.xlsx'


@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    metadata_path = r"C:\Users\lida\Desktop\visymre_10\scripts\weights\meta"
    fitfunc, test_data, _ = vu.setup_visymre_model(cfg,
                                                   metadata_path =metadata_path)

    files = [f for f in os.listdir(ODE_DATA_DIR) if os.path.isfile(os.path.join(ODE_DATA_DIR, f))]
    df_labels = pd.read_excel(ODE_LABELS_FILE)

    results = []

    for i in range(6):  # Run 6 times
        for filename in files:
            print(f"Processing: {filename}")
            try:
                filename_no_ext = os.path.splitext(filename)[0]
                formula_str = df_labels.loc[df_labels['Filename'] == filename_no_ext, 'Formula'].values[0]
            except:
                continue

            filepath = os.path.join(ODE_DATA_DIR, filename)
            data = np.loadtxt(filepath, delimiter=',')

            target_noise = 0.0
            r2_list, solut, pre_expr_list, num_node = [], [], [], []
            start_time = time.time()

            for beam in [30]:
                cfg.inference.beam_size = beam
                r2_list.clear();
                solut.clear();
                pre_expr_list.clear()

                # Split & Noise
                indices = np.random.permutation(len(data))
                split_idx = int(0.75 * len(data))

                # Noise Logic (on y/target which is column 0 here)
                scale = target_noise * np.sqrt(np.mean(np.square(data[indices[:split_idx], 0])))
                noise = np.random.normal(0, scale, data[indices[:split_idx], 0].shape)
                data[indices[:split_idx], 0] += noise

                train_set = data[indices[:split_idx]]
                test_set = data[indices[split_idx:]]

                for _ in range(10):  # Iterations
                    cfg.inference.beam_size = min(30, (_ + 1) * 10)

                    sub_indices = np.random.choice(train_set.shape[0], size=200, replace=False)
                    # ODE specific: y is col 0, X is cols 1:
                    X = torch.tensor(train_set[sub_indices, 1:])
                    y = torch.tensor(train_set[sub_indices, 0].reshape(-1, 1))
                    X_test = torch.tensor(test_set[:, 1:])
                    y_test = torch.tensor(test_set[:, 0])

                    # Pad
                    X = vu.pad_to_10_columns(X)
                    X_test_padded = vu.pad_to_10_columns(X_test)

                    try:
                        output = fitfunc(X, y.squeeze(), cfg_params=cfg.inference, test_data=test_data)
                        pre_expr = sp.sympify(output['best_bfgs_preds'][0])

                        # Eval
                        variables = vu.get_variable_names(str(pre_expr))
                        X_test_dict = {var: X_test_padded[:, idx].cpu() for idx, var in enumerate(variables)}
                        func = sp.lambdify(",".join(variables), pre_expr)
                        y_pre = func(**X_test_dict)

                        sr = vu.symbol_equivalence_single(str(sp.sympify(formula_str)),
                                                          str(vu.coefficient_regularization(str(pre_expr))), variables)

                        r2 = r2_score(y_test, y_pre.real)
                        comp = vu.calculate_tree_size(str(pre_expr))

                    except Exception as e:
                        r2, sr, comp, pre_expr = 0, 0, np.inf, "-"

                    r2_list.append(r2)
                    solut.append(sr)
                    num_node.append(comp)
                    pre_expr_list.append(str(pre_expr))

                    if max(r2_list) > 0.999: break

                if max(r2_list) >= 0.999: break

            elapsed = time.time() - start_time
            best_idx = r2_list.index(max(r2_list))

            results.append([
                filename, formula_str, pre_expr_list[best_idx],
                r2_list[best_idx], solut[best_idx], elapsed,
                num_node[best_idx], target_noise, i
            ])

            pd.DataFrame(results, columns=['filename', 'true_expr', 'pre_expr', 'r2', 'sr', 't', 'Complexity', 'noise',
                                           'Iteration']) \
                .to_csv('visymre_ode_result0.csv', index=False)


if __name__ == "__main__":
    main()