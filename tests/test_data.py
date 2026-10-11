from voyage import data, validate


def test_data_validation_passes():
    validate.validate_data()


def test_route_distance_is_unique_per_route():
    f = data.load_flights()
    assert f.groupby(["from", "to"])["distance"].nunique().max() == 1


def test_user_features_one_row_per_user():
    u, f, h = data.load_users(), data.load_flights(), data.load_hotels()
    df = data.user_behaviour_features(u, f, h)
    assert len(df) == len(u) and df["userCode"].is_unique
    assert not df.drop(columns=["name", "gender", "company"]).isna().any().any()
