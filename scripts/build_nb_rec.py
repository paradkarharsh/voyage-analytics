import sys; sys.path.insert(0, "scripts")
from nbtools import *
nb = new()
header(nb, "Hotel Recommendation", "Recommendation system",
"**Business objective:** suggest hotels to users based on their preferences and booking history (Objective 9), and expose the results in a Streamlit app.")
md(nb, "## ***1. Data & structure***")
code(nb, SETUP + "\nfrom voyage import recommender as rc\nhotels, flights = data.load_hotels(), data.load_flights()\ncat = hotels.groupby('name').agg(place=('place','first'), price=('price','first'), bookings=('total','size')).sort_values('bookings', ascending=False)\ndisplay(cat)\nprint('users:', hotels.userCode.nunique(), '| avg distinct hotels tried per user:', round(hotels.groupby('userCode').name.nunique().mean(), 2), 'of', len(cat))")
md(nb, """Key structure: only **9 hotels**, each in **one city** with **one fixed price**, and users have tried **~7.4 of the 9** on average. So this is not a large-catalogue discovery problem; it is a *ranking* problem over a small set, plus a strong destination signal.""")
md(nb, "## ***2. Approach: hybrid recommender***")
md(nb, """`score(h) = 0.7 x item-CF(h | user history) + 0.3 x popularity(h) + 0.5 x user's own affinity(h)` and, if the next destination is known, a large boost for the hotel in that city.
* **Item-based collaborative filtering:** cosine similarity between hotels' user-booking-count vectors.
* **Popularity:** handles cold-start users (no history).
* **Destination boost / budget filter / exclude-visited:** business rules exposed in the API.""")
code(nb, """rec = rc.HotelRecommender().fit(hotels)
sim = pd.DataFrame(rec.sim_, index=rec.hotels_, columns=rec.hotels_)
sns.heatmap(sim, annot=True, fmt='.2f', cmap='viridis'); plt.title('Hotel-hotel cosine similarity'); plt.show()
print('Off-diagonal similarity range:', round(rec.sim_[rec.sim_>0].min(), 3), '-', round(rec.sim_.max(), 3))""")
md(nb, "Hotel-hotel similarities all fall in a narrow band (**0.57-0.84**) because almost every user tried almost every hotel, so there is little difference between 'similar' and 'dissimilar' hotels. Collaborative filtering therefore has **limited discriminating power** here - an important honest limitation.")
code(nb, """print('Existing user 0:'); display(pd.DataFrame(rec.recommend(0, k=3)))
print('Same user, next trip to Natal (RN):'); display(pd.DataFrame(rec.recommend(0, k=3, destination='Natal (RN)')))
print('Cold-start user, budget <= 150/night:'); display(pd.DataFrame(rec.recommend(None, k=3, max_price=150)))
print('Hotels similar to Hotel A:'); display(pd.DataFrame(rec.similar_hotels('Hotel A')))""")
md(nb, "## ***3. Offline evaluation - predict each user's most recent stay***")
code(nb, """metrics = rc.evaluate_leave_last_out(hotels)
ev = pd.DataFrame({k: v for k, v in metrics.items() if isinstance(v, dict)}).T; display(ev.round(3)); print('users evaluated:', metrics['n_eval_users'])
ev.plot.bar(rot=0); plt.title('Hit rate: is the held-out hotel in the top-k?'); plt.ylabel('hit rate'); plt.show()""")
md(nb, "### Is the improvement over popularity real? (paired bootstrap on hit@3)")
code(nb, """h = hotels.sort_values(['userCode', 'date']); n = h.groupby('userCode').size(); ok = n[n >= 4].index
last = h[h.userCode.isin(ok)].groupby('userCode').tail(1).index; tr, te = h.drop(last), h.loc[last]
r2 = rc.HotelRecommender().fit(tr); pop3 = set(r2.hotel_info_.sort_values('bookings', ascending=False).index[:3])
hy = np.array([row['name'] in [x['hotel'] for x in r2.recommend(row.userCode, k=3)] for _, row in te.iterrows()], float)
po = np.array([row['name'] in pop3 for _, row in te.iterrows()], float)
rng = np.random.default_rng(0); diffs = [(hy[i] - po[i]).mean() for i in (rng.integers(0, len(hy), len(hy)) for _ in range(2000))]
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f'hybrid hit@3 = {hy.mean():.3f}, popularity = {po.mean():.3f}, difference = {hy.mean()-po.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')""")
md(nb, """### Interpretation
Hybrid hit@3 is only marginally above popularity and random (1/3), and the bootstrap interval for the difference includes 0 - **the gain is not statistically established**. The next hotel a user books is close to random given history in this dataset (also seen in the EDA: month, gender and previous hotel explain little).

**Where the system genuinely adds value:** when the destination is known (it is at flight-booking time), the hotel is determined by the city (city <-> hotel is 1:1), so the destination-aware mode is exact by construction; the budget filter and 'similar hotels' features are useful product functions.""")
md(nb, """## ***4. Streamlit application***
`streamlit_app/app.py` (run `make ui`) calls the deployed API and shows: flight price predictor with class comparison, hotel recommender (user / cold-start / destination / budget / exclude-visited) with score charts and the catalogue, travel-insight dashboards, the gender demo with a warning, and a model-metrics page.

## ***Conclusion***
A hybrid recommender is implemented, evaluated with a proper hold-out protocol against popularity and random baselines, and served through the API and dashboard. Modest offline lift is reported honestly.

### ***Hurrah! You have successfully completed the Recommendation part of the capstone!!!***""")
save(nb, "notebooks/04_Hotel_Recommendation.ipynb")
