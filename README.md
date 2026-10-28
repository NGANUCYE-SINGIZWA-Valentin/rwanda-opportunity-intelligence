# Rwanda Opportunity Intelligence

**An interactive district-level explorer and exploratory youth NEET prediction prototype for Rwanda.**

Rwanda Opportunity Intelligence helps government/district planners, youth organizations, researchers, and students explore youth NEET and employment-related indicators. The prototype combines prepared district estimates from the 2025 Labour Force Survey (LFS), separate 2023 Establishment Census context, and an optional locally trained NEET classification model.

> This is a student-built prototype, not an official NISR product. Predictions are exploratory associations, not causes, guarantees, or a forecast of next year's unemployment. Do not use scores to deny services or make high-stakes decisions.

## Features

- National employment and youth NEET overview
- District comparisons and gender comparisons
- Separate Establishment Census 2023 context (establishment counts are not vacancies)
- NEET prediction lab for hypothetical profiles when a locally trained model artifact is available
- Held-out model evaluation metrics and threshold control
- Methodology, limitations, and aggregate CSV download

## Run locally on Windows PowerShell

From the project folder:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Train the NEET model locally

The raw Labour Force Survey microdata is **not included** in this repository. Use the copy you are authorized to access, stored outside the public project folder. From the project root, run one of the following with your actual path:

```powershell
python models\train_neet_model.py --input "C:\path\to\RW_LFS2025.dta"
```

or, if using the official CSV copy:

```powershell
python models\train_neet_model.py --input "C:\path\to\RW_LFS2025.csv"
```

The script uses the supplied codebook's derived variables: it filters to `young == 1` (ages 16–30) and defines NEET as `youngs == 4`. The chosen predictors are sex (`A01`), age (`A04`), marital status (`A05`), area of residence (`Code_UR`), district (`code_dis`), highest education completed (`attained`), disability indicator (`disable`), and household size (`hhsize`). Direct labour-force status, current education/training status, and derived NEET fields are excluded to reduce target leakage.

The script trains a weighted logistic regression using annual weight `weight2`. It holds out entire primary sampling units (PSUs) for the final test, tunes the decision threshold on a separate PSU-level validation split within the training partition, and reports weighted ROC-AUC, average precision, precision, recall, F1, and Brier score. It saves `artifacts/neet_model.joblib` and `artifacts/neet_model_metrics.json` locally. The app detects these files and enables the prediction lab.

### Important evaluation caveats

- A PSU-level split helps reduce cluster leakage, but it is not a complete survey-design variance analysis.
- This is one survey year, so it supports classification of NEET status in survey-like profiles—not a forecast of future unemployment rates.
- Model performance is exploratory and modest. Threshold changes trade off precision and recall. Inspect all reported metrics, especially subgroup performance, before any real-world use.
- The model is not causal and must not be used to deny services, label actual individuals, or make high-stakes decisions.

## Privacy and deployment

- Never add raw `.dta`, raw `.csv` microdata, or `Microdata.zip` to a public GitHub repository.
- The `.gitignore` excludes raw microdata and local `artifacts/` by default.
- **Before deploying the prediction feature**, check the NISR microdata access terms and get confirmation that publishing the trained model artifact and related outputs is permitted. If that is not permitted or unclear, do not publish the model file; deploy the descriptive dashboard and explain that the prediction model is available only in the authorized local environment.
- The aggregate district CSV is also a derived output: confirm its public-release terms before publishing it.

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository and upload the permitted project files.
2. Ensure `app.py`, `requirements.txt`, `README.md`, and the approved aggregate CSV are included.
3. If and only if model publication is permitted, include the trained `artifacts/neet_model.joblib` and `artifacts/neet_model_metrics.json` in the repository (remove the `artifacts/` ignore rule only after confirming permission).
4. Connect the repository at https://share.streamlit.io/, select branch `main`, and set the main file to `app.py`.
5. Test the deployed link in a private/incognito browser and confirm the prediction page behaves as documented.

## Data sources

- National Institute of Statistics of Rwanda (NISR), *Annual Labour Force Survey 2025 Report* and associated variable description file.
- NISR Establishment Census 2023 dataset and accompanying documentation.
- Project-prepared district-level aggregate indicator table.

## Responsible interpretation

District rankings are descriptive and do not establish causes or statistical significance. LFS 2025 indicators and Establishment Census 2023 records are different sources, years, and units. Confirm indicator definitions, confidence intervals, and public-release permissions before policy use.

## AI-use disclosure

Suggested wording—edit it to accurately describe your team's actual use:

> We used an AI assistant to support software scaffolding, code drafting, and documentation. Team members reviewed and tested the application and are responsible for validating all data, calculations, sources, and final claims.

## Hackathon submission checklist

- [ ] Public GitHub repository works without login
- [ ] Deployed app opens without login
- [ ] Data publication terms and trained-model publication terms checked
- [ ] National indicators and definitions verified against NISR source tables
- [ ] Model evaluation and limitations reviewed by both teammates
- [ ] README, sources, and AI-use disclosure updated honestly
- [ ] Repository and deployed-app URLs submitted before the deadline
