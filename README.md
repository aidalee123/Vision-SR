# ViSymRe Benchmark Framework

## 📥 Pre-trained Weights

**Crucial Step:** Before running any scripts, you must download the pre-trained model weights.

1.  **Download Link:** [Get the Pre-trained Weights Here](https://drive.google.com/file/d/1RC1wGPCAe9eTleHC20RGG2c2AqCFG6Af/view?usp=sharing)

## ⚠️ Configuration & Paths (Important)

> **Note: The provided scripts contain hardcoded absolute paths. You must update them to match your local environment before execution.**

**Please manually update the paths in the following files:**

1.  **`config.yaml`**:
    * Update the `model_path` parameter to point to your downloaded `Weight.ckpt`.

2.  **All Python Scripts (`*.py`)**:
    * **Metadata Path**: Open each script and search for `metadata_path`. Update it to point to your local metadata folder:`scripts/weights/meta`.
    * **Dataset Paths**: Refer to the "Dataset Preparation" section below for specific variable names to update in each script.

## 📚 Dataset Preparation

Please download the required datasets (`.txt`, `.tsv`, or `.csv`) from the official repositories or the standardized benchmarks listed below. After downloading, you **must update the file paths** in the corresponding test scripts.

### 1. AI Feynman Dataset
* **Primary Source:** *AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity* (Udrescu et al., 2020).
* **Alternative Source (Standardized):** *Contemporary symbolic regression methods and their relative performance* (La Cava et al., 2021).
* **Download:**
    * [Original AI Feynman Repo](https://github.com/SJ001/AI-Feynman)
    * [SRBench / PMLB Repository](https://github.com/cavalab/srbench).
* **Target Script:** `Feynman_test.py`
* **Configuration:**
    Update the dataset root and label file paths:
```python
# Variable: directory
FEYNMAN_DATA_DIR = '/path/to/your/Feynman_with_units' 

# Variable: directory2
FEYNMAN_LABELS_FILE = 'scripts/FeynmanEquations.xlsx'
```
### 2. SRSD Dataset
* **Source:** *SRSD: Rethinking datasets of symbolic regression for scientific discovery* (Matsubara et al., 2022).
* **Download:** [SRSD Benchmark Repository](https://github.com/omron-sinicx/srsd-benchmark).
* **Target Script:** `SRSD_Feynman_test.py`
* **Configuration:**
    Update the paths for training, testing, and ground truth expressions:
```python
# Variable: directory
TRAIN_SET_DIR = "/path/to/srsd_benchmark/*/train"

# Variable: directory2
TEST_SET_DIR = "/path/to/srsd_benchmark/*/test"

# Variable: directory3
GROUND_TRUTH_DIR = "/path/to/srsd_benchmark/*/true_expr"
```
### 3. ODE-Strogatz Dataset
* **Primary Source:** *Nonlinear dynamics and chaos* (Strogatz, 2018).
* **Digital Source (Standardized):** *Contemporary symbolic regression methods and their relative performance* (La Cava et al., 2021).
* **Download:** Please acquire the Strogatz ODE benchmark dataset, commonly available via the [SRBench Repository](https://github.com/cavalab/srbench).
* **Target Script:** `ODE-Strogatz_test.py`
* **Configuration:**
    Update the dataset directory and label file:
```python
# Variable: directory
ODE_DATA_DIR = '/path/to/ode-strogatz-master'

# Variable: directory2
ODE_LABELS_FILE = 'scripts/ode.xlsx'
```
### 4. SRBench / Black-box Datasets
* **Source:** *Call for Action: towards the next generation of symbolic regression benchmark* (Aldeia et al., 2025).
* **Download:** [SRBench / PMLB Repository](https://github.com/cavalab/srbench) (Look for "Black-box" regression datasets).
* **Target Script:** `Black-box_test.py`
* **Configuration:**
    Update the local data directory:
```python
# Variable: LOCAL_DATA_DIR
BLACKBOX_DATA_DIR = "/path/to/srbench_blackbox_datasets"
```
### 5. First Principles Datasets
* **Source:** *Contemporary symbolic regression methods and their relative performance* (La Cava et al., 2021).
* **Download:** [SRBench / PMLB Repository](https://github.com/cavalab/srbench) (Look for "First Principles" datasets, e.g., `first_principles_bode`, etc.).
* **Target Script:** `First_Principles_test.py`
* **Configuration:**
    Update the root path for First Principles datasets:
```python
# Variable: DATASETS_ROOT
FIRST_PRINCIPLES_ROOT = "/path/to/your/downloaded/firstprinciples_data"
```
### 6. Low-dimensional_benchmark_test

* **Configuration:**
 * **Target Script:** `Low-dimensional_benchmark_test.py`
```python
# Variable: DATASETS_ROOT
FIRST_PRINCIPLES_ROOT = "scripts/low_benchmarks.csv"
```
### 7. Scale_ablation_test
* **Configuration:**
 * **Target Script:** `Scale_ablation_test.py`
```python
# Variable: DATASETS_ROOT
FIRST_PRINCIPLES_ROOT = "low_benchmarks_scale.csv"
```
**References:**
1.  Udrescu, S. M., et al. (2020). "AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity". *Advances in Neural Information Processing Systems*.
2.  Strogatz, S. H. (2018). *Nonlinear dynamics and chaos: with applications to physics, biology, chemistry, and engineering*. CRC press.
3.  Matsubara, Y., et al. (2022). "SRSD: Rethinking datasets of symbolic regression for scientific discovery". *NeurIPS 2022 AI for Science: Progress and Promises*.
4.  La Cava, W., et al. (2021). "Contemporary symbolic regression methods and their relative performance". *arXiv preprint arXiv:2107.14351*.
5.  Aldeia, G. S. I., et al. (2025). "Call for Action: towards the next generation of symbolic regression benchmark". *arXiv preprint arXiv:2505.03977*.