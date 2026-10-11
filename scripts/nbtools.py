"""Tiny helpers for generating notebooks programmatically."""
import nbformat as nbf

def new(): return nbf.v4.new_notebook()
def md(nb, text): nb.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
def code(nb, text): nb.cells.append(nbf.v4.new_code_cell(text.strip("\n")))
def save(nb, path):
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, path)

SETUP = '''
import sys, warnings, pathlib
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from voyage import config, data
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.figsize": (9, 4.5), "axes.titleweight": "bold"})
pd.set_option("display.max_columns", 30)
'''

def header(nb, title, typ, extra_problem):
    md(nb, f"""# **Project Name** - **{title}**
##### **Project Type** - {typ}
##### **Contribution** - Team (3 members)

# **Problem Statement**
**Voyage Analytics** is a travel-tech capstone built on three datasets - *users*, *flights* and *hotels* - with a dual goal:
(1) use them to make better travel decisions with machine learning, and (2) learn MLOps by productionising those models
(REST API -> Docker -> Kubernetes -> Airflow -> Jenkins CI/CD -> MLflow -> Streamlit).

{extra_problem}""")
