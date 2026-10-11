import sys; sys.path.insert(0, "scripts")
from nbtools import *
nb = new()
header(nb, "Gender Classification", "Classification (3-class: female / male / none)",
"**Business objective:** deploy a classifier that categorises a user's gender label from demographics and travel behaviour (Objective 8). A key part of the task is judging **whether the data supports such a model at all**.")
md(nb, "## ***1. Build the user-level dataset***")
code(nb, SETUP + "\nfrom voyage import train_gender as tg\nusers, flights, hotels = data.load_users(), data.load_flights(), data.load_hotels()\ndf = data.user_behaviour_features(users, flights, hotels)\nprint(df.shape); display(df[['gender'] + tg.FEATURES].head())\nprint(df.gender.value_counts(normalize=True).round(3).to_dict())")
md(nb, """One row per user (1,340). Features: `age`, `company`, plus behaviour aggregated from flights (count, average price, distance, class mix, agencies, destinations) and hotels (stays, average length, average spend, distinct hotels). The classes are balanced, so **chance accuracy is ~33%** and the majority-class baseline is ~34%. Note that 'none' is a genuine third label, not missing data.""")
md(nb, "## ***2. Do the features carry any signal? (before modelling)***")
code(nb, """from scipy import stats
rows = []
for c in tg.NUMERIC:
    p = stats.f_oneway(*[g[c] for _, g in df.groupby('gender')]).pvalue; rows.append((c, p))
for c in tg.CATEGORICAL:
    rows.append((c, stats.chi2_contingency(pd.crosstab(df[c], df.gender))[1]))
sig = pd.DataFrame(rows, columns=['feature', 'p_value']).sort_values('p_value').round(4)
sig['significant_at_5%'] = sig.p_value < .05
display(sig)
print(f"Expected false positives at 5% with {len(sig)} tests: {0.05*len(sig):.1f}")""")
md(nb, "If a feature separated the classes, its p-value would be tiny. Only `avg_stay_days` falls below 0.05 (p ≈ 0.019). With 13 tests, ~0.7 false positives are expected by chance, and after a Bonferroni correction (threshold 0.05/13 ≈ 0.004) nothing is significant - so this is best treated as a chance finding (and a user's average stay length is not a plausible driver of gender).")
md(nb, "## ***3. Models (5-fold stratified CV on the training split + hold-out test)***")
code(nb, """from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score
X, y = df[tg.FEATURES], df.gender
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows, fitted = [], {}
for name, est in tg.candidate_models().items():
    pipe = tg.make_pipeline(est); r = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=['accuracy', 'f1_macro'])
    pipe.fit(Xtr, ytr); p = pipe.predict(Xte); fitted[name] = pipe
    rows.append({'model': name, 'cv_accuracy': r['test_accuracy'].mean(), 'cv_accuracy_std': r['test_accuracy'].std(),
                 'cv_f1_macro': r['test_f1_macro'].mean(), 'test_accuracy': accuracy_score(yte, p), 'test_f1_macro': f1_score(yte, p, average='macro')})
res = pd.DataFrame(rows).set_index('model').round(3); display(res)
print('Hold-out test set size:', len(yte), '-> one standard error of accuracy ~', round(float(np.sqrt(.33*.67/len(yte))), 3))""")
md(nb, """### Interpretation
All models sit within roughly **±3 points of 33%**. The hold-out test accuracies (~34-36%) look slightly higher than CV (~32-33%), but with only 268 test users one standard error is ~2.9 points, so that gap is noise - **CV is the more trustworthy number**. None of the models is convincingly better than the majority baseline.""")
md(nb, "## ***4. Is the result statistically different from guessing? (permutation test)***")
code(nb, """from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LogisticRegression
score, perm, pval = permutation_test_score(tg.make_pipeline(LogisticRegression(max_iter=2000, class_weight='balanced')), X, y,
                                          cv=cv, scoring='accuracy', n_permutations=100, random_state=0, n_jobs=1)
print(f'real CV accuracy = {score:.3f} | mean accuracy with shuffled labels = {perm.mean():.3f} | permutation p-value = {pval:.3f}')
plt.hist(perm, bins=20, alpha=.7, label='shuffled labels'); plt.axvline(score, color='r', label='real labels'); plt.legend(); plt.title('Permutation test'); plt.show()""")
md(nb, "If the real-label score sits inside the shuffled-label distribution (large p-value), the model has learned nothing beyond chance.")
code(nb, """from sklearn.metrics import ConfusionMatrixDisplay, classification_report
best = fitted['logistic_regression']
ConfusionMatrixDisplay.from_estimator(best, Xte, yte, cmap='Blues'); plt.title('Confusion matrix (logistic regression, test)'); plt.show()
print(classification_report(yte, best.predict(Xte)))""")
md(nb, """## ***5. Conclusion & responsible-use note***
* **Result:** the gender label cannot be predicted from age, company or travel behaviour in this dataset (EDA hypothesis tests H3/H4, the per-feature tests above and the permutation test agree).
* **Deployment:** the model is still packaged and served (`POST /predict/gender`) to demonstrate the full path - but the API returns calibrated class probabilities plus an explicit low-confidence warning, and the MLflow run records `beats_majority_baseline`.
* **What to say in the viva:** a data scientist's job includes reporting a null result honestly. Do not present ~34% accuracy as success.
* **Ethics:** inferring gender from behaviour is sensitive; in a real product it should be based on self-declared data, with consent, not predicted.
* **If you wanted to improve it:** you would need genuinely informative features (e.g. booking text, preferences) - not more tuning of these ones.

### ***Hurrah! You have successfully completed the Classification part of the capstone!!!***""")
save(nb, "notebooks/03_Gender_Classification.ipynb")
