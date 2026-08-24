markdown
# Brite Spark 2026 — Overpayment Signal

## Step-by-Step Setup (Windows / VS Code)

### Step 1 — Open the project
Open the project folder in VS Code and open the terminal (Terminal > New Terminal).

### Step 2 — Go to the project folder (Auto-Find)
> **This prevents the "No such file" error.** Paste this command to automatically find the folder containing `main.py` (even if it's nested inside another folder):

```powershell
cd (Get-ChildItem -Recurse -Filter main.py -Depth 5 -ErrorAction SilentlyContinue | Select-Object -First 1).Directory.FullName

Step 3 — Create virtual environment
powershell
python -m venv venv

Step 4 — Allow PowerShell activation
powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Step 5 — Activate virtual environment
powershell
.\venv\Scripts\Activate.ps1

Step 6 — Install dependencies
powershell
python -m pip install -r requirements.txt

Step 7 — Run the project
powershell
python main.py

Step 8 — Check results
Results are generated in the output/ folder.
View the fairness report (most important):

powershell
Get-Content output\fairness_report.txt
