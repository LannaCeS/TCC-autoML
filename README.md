# TCC-autoML

## Python Environments

The notebooks use three local Python environments. Create them from the repository root. Make sure Python 3.10, 3.11, and 3.12 are installed and available as `python3.10`, `python3.11`, and `python3.12`.

```bash
python3.12 -m venv venv-autogluon
python3.10 -m venv venv10
python3.11 -m venv venv11

./venv-autogluon/bin/python -m pip install --upgrade pip ipykernel autogluon
./venv10/bin/python -m pip install --upgrade pip ipykernel
./venv11/bin/python -m pip install --upgrade pip ipykernel

./venv-autogluon/bin/python -m ipykernel install --user --name venv-autogluon --display-name "Python (venv-autogluon)"
./venv10/bin/python -m ipykernel install --user --name venv10 --display-name "Python (venv10, 3.10)"
./venv11/bin/python -m ipykernel install --user --name venv11 --display-name "Python (venv11, 3.11)"
```

The AutoGluon environment uses Python 3.12, matching the saved kernel version in the AutoGluon notebooks. Keep it separate from the 3.10 and 3.11 environments because AutoML packages have conflicting Python and scikit-learn requirements. Activate an environment in a terminal with `source venv-autogluon/bin/activate`, `source venv10/bin/activate`, or `source venv11/bin/activate`. In Jupyter, select the matching registered kernel.

## Notebook Kernel Mapping

In VS Code, open one of the notebooks, click **Select Kernel** in the top-right corner, choose **Python Environments**, then select the environment as shown below. Restart the kernel after switching. In JupyterLab, use **Kernel > Change Kernel**. Use `%pip install ...` in a notebook to add a missing package to its selected environment, then restart the kernel.

| Notebook | Environment |
| --- | --- |
| `notebooks/Autogluon.ipynb` | `venv-autogluon` |
| `notebooks/balanced/Autogluon.ipynb` | `venv-autogluon` |
| `notebooks/AutoMLs 1.ipynb` | base legacy notebook, not for running experiments |
| `credit_card_fraud_pipeline_modified_last_ensemble_28082025v12.ipynb` | `venv-autogluon` (Python 3.12); run its package-install cells for the pipeline dependencies |
| `notebooks/Needs_venv10.ipynb` | `venv10` |
| `notebooks/balanced/Needs_venv10.ipynb` | `venv10` |
| `AutoMLs_1.ipynb` |  base legacy notebook, not for running experiments |
| `notebooks/Needs_venv11.ipynb` | `venv11` |
| `notebooks/balanced/Needs_venv11.ipynb` | `venv11` |
| `notebooks/AutoMLs2.ipynb` | `venv10` for compatible sections; use the Docker services below for AutoPyTorch and Auto-sklearn |

Some notebooks contain commented-out install cells. Uncomment and run only the package installs needed for the sections being executed. The TPOT notebooks use Python 3.10; the FLAML, H2O, LazyPredict, and PyCaret notebooks use Python 3.11. AutoGluon install cells can be skipped after it is installed in `venv-autogluon`.

## Legacy AutoML Services

The repository includes Docker services for packages that are difficult to install in the local macOS environments, notably AutoPyTorch and Auto-sklearn. These services run their own JupyterLab instances; open the notebook in the matching service and run it there rather than selecting a local venv kernel:

```bash
docker compose up auto-pytorch  # http://localhost:8882
docker compose up auto-sklearn  # http://localhost:8881
docker compose up lazypredict   # http://localhost:8883
```

AutoPyTorch's Docker image uses Python 3.8 and pinned legacy dependencies, so it is intentionally not part of `venv10` or `venv11`. On Apple Silicon, the AutoPyTorch and Auto-sklearn services use the configured `linux/amd64` platform and may run under emulation.