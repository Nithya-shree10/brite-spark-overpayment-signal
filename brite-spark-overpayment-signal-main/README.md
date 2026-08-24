# Problem 6: The Overpayment Signal

## Brite Spark 2026

A machine learning solution for ranking suspicious benefit payments using unsupervised anomaly detection with fairness analysis.

---

## Quick Start (How to Run)

### Step 1: Navigate to the project folder
Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and go into the project folder:

```bash
cd 06-overpayment-signal

Step 2: Set up a virtual environment (Recommended)
bash
python -m venv venv

On Windows:
bash
venv\Scripts\activate

On Mac/Linux:

bash
source venv/bin/activate

Step 3: Install dependencies
bash
python -m pip install --upgrade pip

bash
python main.py

Step 5: View the results

On Windows PowerShell:
powershell
Get-Content output\fairness_report.txt
Get-Content output\model_limitations.txt

On Mac/Linux:
bash
cat output/fairness_report.txt
cat output/model_limitations.txt
