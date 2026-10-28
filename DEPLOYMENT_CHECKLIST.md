# Deployment checklist

## Before pushing to GitHub
- [ ] Remove raw microdata from the project folder if you copied it there.
- [ ] Confirm the repository does not contain `.dta`, raw LFS `.csv`, `Microdata.zip`, or household/person records.
- [ ] Confirm NISR terms permit public release of the prepared district aggregates.
- [ ] Confirm NISR terms permit public release of the trained model artifact before removing the `artifacts/` ignore rule.
- [ ] Check README and AI-use disclosure against what the team actually did.

## Local test
- [ ] `pip install -r requirements.txt` succeeds.
- [ ] `streamlit run app.py` starts without errors.
- [ ] Dashboard overview and district filters work.
- [ ] If trained locally, prediction lab shows metrics and returns a hypothetical-profile estimate.
- [ ] Model metrics include PSU-level holdout and the threshold-selection method.
- [ ] No claim describes the classifier as a future trend forecast or a causal model.

## GitHub and deployment
- [ ] Public repository contains `app.py`, `requirements.txt`, `README.md`, `backend/`, `models/`, and approved aggregate data.
- [ ] Include `artifacts/` only if publishing the trained model is permitted.
- [ ] Streamlit Community Cloud deploys from `main` and `app.py`.
- [ ] Test deployed link in a private/incognito browser.
- [ ] If no model artifact is published, ensure the app clearly explains the prediction feature is unavailable in the public deployment.
- [ ] Submit GitHub link, deployed app link, and documentation before the deadline.
