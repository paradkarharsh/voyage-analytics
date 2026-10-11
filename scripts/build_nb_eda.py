import sys; sys.path.insert(0, "scripts")
from nbtools import *
nb = new()
header(nb, "Voyage Analytics - Exploratory Data Analysis", "EDA + Hypothesis Testing",
"""**Business objective of this notebook:** understand who travels, where, on what class/agency, what they pay, and where they
stay - and find out which variables can legitimately drive the prediction models built in notebooks 02-04.""")
md(nb, "## ***1. Know Your Data***\n### Import Libraries & Load Data")
code(nb, SETUP)
code(nb, "users, flights, hotels = data.load_users(), data.load_flights(), data.load_hotels()\nfor n, d in [('users', users), ('flights', flights), ('hotels', hotels)]:\n    print(f'--- {n}: {d.shape[0]:,} rows x {d.shape[1]} cols'); display(d.head(3))")
md(nb, "### Dataset Information, Duplicates, Missing Values")
code(nb, """for n, d in [('users', users), ('flights', flights), ('hotels', hotels)]:
    print(f"{n:8s} duplicates={d.duplicated().sum()}  missing cells={int(d.isna().sum().sum())}")
print('\\nFlight dates :', flights.date.min().date(), '->', flights.date.max().date())
print('Hotel dates  :', hotels.date.min().date(), '->', hotels.date.max().date())
flights.info()""")
md(nb, """### What did you know about your dataset?
* **users** (1,340) - `code`, `company`, `name`, `gender` (male / female / **none**), `age` (21-65).
* **flights** (271,888) - one row per flight leg: route (`from`, `to`), `flightType` (economic / premium / firstClass), `price`, `time`, `distance`, `agency`, `date`.
* **hotels** (40,552) - one row per stay: `name`, `place`, `days`, `price` (per night), `total`, `date`.
* **No missing values and no duplicates** - the data is clean; the challenge is in what the data *cannot* tell us (see the findings below).
* `travelCode` + `userCode` link a trip's flights and hotel stay; `userCode` links everything to `users.code`.""")
md(nb, "## ***2. Understanding Your Variables***")
code(nb, "display(flights.describe().round(2)); display(hotels.describe().round(2)); display(users.describe().round(2))\nprint(flights[['from','to','flightType','agency']].nunique())")
md(nb, "## ***3. Data Wrangling***")
code(nb, """# Integrity checks that matter for modelling
print('Distinct distances per route (should be 1):', flights.groupby(['from','to']).distance.nunique().max())
print('|time/distance| spread (constant speed?):', (flights.time/flights.distance).agg(['min','max']).round(5).to_dict())
print('hotel total == days*price for all rows :', bool(((hotels.days*hotels.price - hotels.total).abs() < 0.05).all()))
print('Distinct price per hotel               :', hotels.groupby('name').price.nunique().max())
print('Hotel city == flight destination (same trip):',
      round((hotels.merge(flights.sort_values('date').groupby(['travelCode','userCode']).head(1)[['travelCode','userCode','to']],
             on=['travelCode','userCode']).eval('place == to')).mean(), 3))
# Is price a pure function of (route, class, agency)?
g = flights.groupby(['from','to','flightType','agency']).price.std()
print('Max price std within (route, class, agency):', round(float(g.max()), 4), f'over {len(g)} combinations')""")
md(nb, """### What manipulations have you done and insights you found?
Four structural facts fall out of the integrity checks - each one drives a modelling decision later:
1. **`time` is a constant multiple of `distance`** (correlation 1.000) -> keep `distance`, drop `time` to avoid perfect multicollinearity.
2. **Each route has exactly one distance**, so an API can accept just `from`/`to` and look the distance up server-side.
3. **Flight price is fully determined by (route, flightType, agency)**: the standard deviation within every one of the 490 combinations is 0. Prices are a fixed price table, so a regression model can reach R² ≈ 1 legitimately (this is *not* leakage - no target-derived column is used) but it also means dates add nothing.
4. **Each hotel is in exactly one city with one fixed nightly price, and the hotel city always equals the flight destination of the same trip** - that makes destination the strongest recommendation signal.""")
md(nb, "## ***4. Data Visualization, Storytelling & Experimenting with charts***")

def chart(n, title, code_txt, why, insight, impact):
    md(nb, f"#### Chart - {n} - {title}")
    code(nb, code_txt)
    md(nb, f"##### 1. Why did you pick the specific chart?\n{why}\n##### 2. What is/are the insight(s) found from the chart?\n{insight}\n##### 3. Will the gained insights help creating a positive business impact?\n{impact}")

chart(1, "Flight class mix (Univariate)",
"""c = flights.flightType.value_counts()
ax = c.plot.bar(color=sns.color_palette()[:3]); ax.set_ylabel('flights'); ax.set_title('Flights by class')
for i, v in enumerate(c): ax.text(i, v, f'{v:,}\\n({v/len(flights):.0%})', ha='center', va='bottom')
plt.xticks(rotation=0); plt.ylim(0, c.max()*1.2); plt.show()""",
"A bar chart is the clearest way to compare the size of a handful of categories.",
"First class is the largest segment (**116,418 flights, 43%**), ahead of premium (78,004; 29%) and economic (77,466; 28%). This is an unusually premium-heavy customer base.",
"Yes - revenue is concentrated in first class, so pricing, capacity and upsell strategy should be built around it; a price model must be very accurate on first class.")
chart(2, "Price distribution by class (Bivariate: categorical vs numeric)",
"sns.boxplot(data=flights, x='flightType', y='price', order=['economic','premium','firstClass']); plt.title('Flight price by class'); plt.show()\nprint(flights.groupby('flightType').price.agg(['mean','median','std']).round(0))",
"Box plots show median, spread and overlap between groups at once.",
"Mean price rises cleanly with class: **economic ≈ 658, premium ≈ 920, first class ≈ 1,181**. The classes overlap substantially, so class alone does not fix the price - route matters too.",
"Yes - a clear class premium (~1.8x economic for first class) supports tiered pricing and is a key feature for the regression model.")
chart(3, "Agency price - the confounding trap (Bivariate + Multivariate)",
"""fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
flights.groupby('agency').price.mean().plot.bar(ax=ax[0], color=sns.color_palette()[:3], rot=0); ax[0].set_title('Average price by agency (naive)')
flights.groupby(['flightType','agency']).price.mean().unstack().plot.bar(ax=ax[1], rot=0); ax[1].set_title('Average price by class AND agency')
plt.tight_layout(); plt.show()
print(flights.groupby('agency').flightType.value_counts(normalize=True).unstack().round(2))""",
"Placing the naive view next to the class-split view exposes a classic Simpson's-paradox style confound.",
"Naively **FlyingDrops looks ~30% more expensive (≈1,186 vs ≈919)**. But **FlyingDrops sells *only* first-class tickets** while CloudFy and Rainbow sell a 1/3-1/3-1/3 mix of all classes. Within first class the three agencies charge nearly the same (≈1,177-1,186, a gap under 1%).",
"Yes - it prevents a wrong business conclusion. Agencies differ mainly in product mix, not in price. Any agency recommendation must control for class.")
chart(4, "Destinations - volume and price (Univariate + Bivariate)",
"""fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
flights.groupby('to').size().sort_values().plot.barh(ax=ax[0]); ax[0].set_title('Flights by destination')
flights.groupby('to').price.mean().sort_values().plot.barh(ax=ax[1], color='tab:orange'); ax[1].set_title('Average price by destination')
plt.tight_layout(); plt.show()""",
"Horizontal bars keep the long city names readable and rank cities at a glance.",
"**Florianopolis is the top destination (57,317 flights, 21%)**, followed by Aracaju (37,224) and Campo Grande (34,748). **Salvador is the most expensive (≈1,264 avg) and Sao Paulo the cheapest (≈648).**",
"Yes - capacity planning and marketing should prioritise Florianopolis; Salvador routes carry the highest per-ticket revenue.")
chart(5, "Price vs distance (Bivariate: numeric vs numeric)",
"""s = flights.sample(15000, random_state=1)
sns.scatterplot(data=s, x='distance', y='price', hue='flightType', alpha=.4, s=12); plt.title('Price vs distance'); plt.show()
print(flights[['price','distance','time']].corr().round(3))""",
"A scatter plot reveals the shape of a relationship and colour reveals the class structure.",
"Price rises with distance (**r = 0.64**) but forms three separate class bands. `time` and `distance` are perfectly correlated (**r = 1.00**) - redundant.",
"Yes - distance is a legitimate driver of price and `time` can be dropped from the model with zero information loss.")
chart(6, "Demand over time (Time series)",
"""m = flights.set_index('date').resample('MS').size()
m.plot(); plt.title('Flights per month'); plt.ylabel('flights'); plt.show()
print(flights.groupby(flights.date.dt.year).size())""",
"A line chart is the standard way to see trend and seasonality.",
"The window runs **26 Sep 2019 -> 24 Jul 2023**. Volume peaks in **2020 (112,571 flights)** and declines afterwards (2021: 75,363; 2022: 41,761; 2023 partial: 6,367). The pattern does not follow real-world travel seasonality, which is one of the signs that this is a synthetic dataset - we treat time effects with caution.",
"Partly - trend monitoring is useful for MLOps drift checks, but forecasting from this pattern would not be credible.")
chart(7, "Hotels - bookings vs revenue (Bivariate)",
"""hb = hotels.groupby('name').agg(bookings=('total','size'), price=('price','first'), revenue=('total','sum')).sort_values('revenue')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
hb.bookings.plot.barh(ax=ax[0]); ax[0].set_title('Bookings per hotel')
hb.revenue.plot.barh(ax=ax[1], color='tab:green'); ax[1].set_title('Total revenue per hotel')
plt.tight_layout(); plt.show(); display(hb.round(0))""",
"Two aligned bar charts contrast popularity (volume) with value (revenue).",
"**Hotel K (5,094) and Hotel CB (5,029) are the most booked**, but **Hotel AU earns the most (≈3.52M)** thanks to a ≈313/night rate. **Hotel BW is the cheapest (≈60/night) and lowest revenue (≈0.65M)**. Hotel A is the least booked (3,330).",
"Yes - popularity and revenue rank hotels differently, so a recommender should be able to weigh price/budget, not just popularity.")
chart(8, "Length of stay and customers (Univariate)",
"""fig, ax = plt.subplots(1, 3, figsize=(15, 4))
hotels.days.value_counts().sort_index().plot.bar(ax=ax[0], rot=0); ax[0].set_title('Nights per stay')
sns.histplot(users.age, bins=20, ax=ax[1]); ax[1].set_title('Customer age')
users.gender.value_counts().plot.bar(ax=ax[2], rot=0); ax[2].set_title('Gender labels')
plt.tight_layout(); plt.show(); print(users.company.value_counts())""",
"Small-multiple charts profile the three key univariate distributions side by side.",
"Stays are **exactly uniform: 25% each for 1-4 nights**. Customers are aged **21-65 (mean ≈ 42.7)**. Gender labels are almost perfectly balanced: **male 452, female 448, none 440**; 4You is the largest company (453 users).",
"Yes - a uniform stay length and balanced gender classes mean no single 'typical' stay to optimise, and balanced classes make gender modelling fair to evaluate (chance = 33%).")
chart(9, "Correlation heatmap (Multivariate)",
"""uf = data.user_behaviour_features(users, flights, hotels)
num = uf.select_dtypes('number').drop(columns='userCode')
plt.figure(figsize=(9, 7)); sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, annot_kws={'size': 7}); plt.title('User-level feature correlations'); plt.show()""",
"A heatmap summarises all pairwise linear relationships in one view.",
"Behavioural features are largely uncorrelated with `age`, and per-user averages (flight price, spend) barely relate to each other - users behave like independent random draws from the same population.",
"Yes - it warns us early that user-level prediction tasks (like gender) will have very little signal to learn from.")

md(nb, "## ***5. Hypothesis Testing***\nFour statements were derived from the charts above.")
code(nb, """from scipy import stats
# H1: first class costs more than economic  (one-tailed Mann-Whitney U on equal-size samples)
fc = flights[flights.flightType=='firstClass'].price.sample(20000, random_state=1)
ec = flights[flights.flightType=='economic'].price.sample(20000, random_state=1)
print('H1  first > economic  : p =', stats.mannwhitneyu(fc, ec, alternative='greater').pvalue)

# H2: within first class, the three agencies charge the same average price (one-way ANOVA)
fcl = flights[flights.flightType=='firstClass']
print('H2  agency (1st class): p =', round(stats.f_oneway(*[g.price for _, g in fcl.groupby('agency')]).pvalue, 4),
      '| means', fcl.groupby('agency').price.mean().round(1).to_dict())

# H3: average hotel spend is the same across the 3 gender labels (ANOVA, user level)
print('H3  gender vs spend   : p =', round(stats.f_oneway(*[g.avg_hotel_spend for _, g in uf.groupby('gender')]).pvalue, 4))

# H4: gender is independent of company (chi-square)
print('H4  gender x company  : p =', round(stats.chi2_contingency(pd.crosstab(uf.company, uf.gender))[1], 4))""")
md(nb, """| # | Null hypothesis (H0) | Alternate (H1) | Test | Result |
|---|---|---|---|---|
| H1 | mean first-class price = economic | first class > economic | Mann-Whitney U (right-tailed; skew-robust) | **Reject H0** (p ≈ 0) - first class is genuinely pricier |
| H2 | agencies charge the same within first class | at least one differs | One-way ANOVA | **Reject H0** (p ≈ 0.0007) - but the gap is only ≈ 9 (< 1%), i.e. statistically detectable because n is huge, **not practically meaningful** |
| H3 | hotel spend equal across gender labels | differs | One-way ANOVA | **Fail to reject** (p ≈ 0.26) |
| H4 | gender independent of company | dependent | Chi-square | **Fail to reject** (p ≈ 0.37) |

H3 and H4 are an early warning that **gender is not related to spending behaviour or employer** in this data.""")
md(nb, """## ***6. Solution to Business Objective & Conclusion***
1. **Prices are a price table** - route + class + agency fully determine the fare. The regression model is therefore legitimately near-perfect; dates and `time` add nothing.
2. **Agency 'premium' is a mix effect** - FlyingDrops only sells first class. Compare agencies within a class (differences there are < 1%).
3. **Florianopolis / first class / Hotel AU** are the volume and value anchors of the business.
4. **Hotel city == flight destination** - the recommender should use the destination whenever it is known.
5. **User attributes (gender/age/company) are essentially independent of behaviour** - so the gender classifier should be expected to perform near chance (verified in notebook 03).
6. The data window and volume pattern look **synthetic**; use it to demonstrate the MLOps pipeline, and be cautious about real-world claims.

### ***Hurrah! You have successfully completed the EDA part of the capstone!!!***""")
save(nb, "notebooks/01_EDA_Voyage_Analytics.ipynb")
