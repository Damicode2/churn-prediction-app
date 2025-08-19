git init
git add .
git commit -m "Initial commit - churn prediction app"
git branch -M main
git remote add origin https://github.com/Damicode2/churn-prediction-app.git
git push -u origin main
git add app.py requirements.txt
git commit -m "fix: add XGBoost import for pipeline unpickling"
git push
