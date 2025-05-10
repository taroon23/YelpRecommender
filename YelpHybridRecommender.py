# ------------------- Library Imports -------------------
import sys
import csv
import json
import math
import heapq
import random
import xgboost as xgb
from pyspark import SparkContext, SparkConf

# ------------------- Collaborative Filtering (CF) Logic -------------------
# Compute Pearson similarity between two businesses
def pearson_similarity(b1_ratings, b2_ratings, user_avg_dict, global_avg, shrinkage=10):
    common_users = set(b1_ratings.keys()) & set(b2_ratings.keys())
    if len(common_users) < 5:
        return 0
    # Center ratings around user mean
    b1_vals = [b1_ratings[u] - user_avg_dict.get(u, global_avg) for u in common_users]
    b2_vals = [b2_ratings[u] - user_avg_dict.get(u, global_avg) for u in common_users]
    num = sum(a * b for a, b in zip(b1_vals, b2_vals))
    denom1 = math.sqrt(sum(a ** 2 for a in b1_vals))
    denom2 = math.sqrt(sum(b ** 2 for b in b2_vals))
    if denom1 == 0 or denom2 == 0:
        return 0
    raw_sim = num / (denom1 * denom2)
    return raw_sim * (len(common_users) / (len(common_users) + shrinkage))

# Predict CF-based rating for a given user-business pair
def predict_cf(user, business, user_ratings_dict, business_ratings_dict, user_avg_dict, business_avg_dict, global_avg):
    if user not in user_ratings_dict and business not in business_ratings_dict:
        return global_avg
    if business not in business_ratings_dict:
        return user_avg_dict.get(user, global_avg)
    if user not in user_ratings_dict:
        return business_avg_dict.get(business, global_avg)

    # Get similar businesses user has rated
    target_ratings = business_ratings_dict[business]
    user_rated = user_ratings_dict[user]
    N = min(80, max(25, int(0.25 * len(user_rated))))

    heap = []
    for b, r in user_rated.items():
        if b == business or b not in business_ratings_dict:
            continue
        sim = pearson_similarity(target_ratings, business_ratings_dict[b], user_avg_dict, global_avg)
        if sim > 0.2:
            heapq.heappush(heap, (-sim, sim, r))
            if len(heap) > N:
                heapq.heappop(heap)

    # Compute weighted average prediction
    b_avg = business_avg_dict.get(business, global_avg)
    u_avg = user_avg_dict.get(user, global_avg)
    baseline = 0.5 * b_avg + 0.4 * u_avg + 0.1 * global_avg

    if not heap:
        return baseline

    weighted_sum = sum(sim * r for _, sim, r in heap)
    sim_sum = sum(abs(sim) for _, sim, _ in heap)
    if sim_sum == 0:
        return baseline

    pred = weighted_sum / sim_sum
    return min(5.0, max(1.0, 0.85 * pred + 0.15 * baseline))

# ------------------- Feature Extraction Helpers -------------------
# Safely extract boolean-like attributes
def safe_bool(attrs, key):
    return 0 if (attrs is None or attrs.get(key, 'False') == 'False') else 1

# Estimate or retrieve price range
def price_range(attrs):
    if attrs is None: return random.randint(1, 4)
    pr = attrs.get('RestaurantsPriceRange2')
    return int(pr) if pr and pr.isdigit() else random.randint(1, 4)

# Count elite years
def user_elite_cnt(elite_str): return 0 if elite_str == 'None' else len(elite_str.split(','))

# Count number of friends
def user_friends_cnt(f_str): return 0 if f_str == 'None' else len(f_str.split(','))

# Parse business JSON data into structured tuple
def parse_business(b):
    attrs = b.get('attributes')
    return (b["business_id"],
            (b.get("stars", 3.5),
             b.get("review_count", 50),
             b.get("latitude", 0),
             b.get("longitude", 0),
             price_range(attrs),
             safe_bool(attrs, 'BusinessAcceptsCreditCards'),
             safe_bool(attrs, 'ByAppointmentOnly'),
             safe_bool(attrs, 'RestaurantsReservations'),
             safe_bool(attrs, 'RestaurantsTableService'),
             safe_bool(attrs, 'WheelchairAccessible')))

# Parse user JSON data into structured tuple
def parse_user(u):
    comps = [u.get(k, 0) for k in ['compliment_hot', 'compliment_profile', 'compliment_list',
                                   'compliment_note', 'compliment_plain', 'compliment_cool',
                                   'compliment_funny', 'compliment_writer', 'compliment_photos']]
    return (u['user_id'],
            (u.get('review_count', 10),
             user_friends_cnt(u.get('friends', 'None')),
             u.get('useful', 2),
             u.get('funny', 0),
             u.get('cool', 0),
             u.get('fans', 0),
             user_elite_cnt(u.get('elite', 'None')),
             u.get('average_stars', 3.5)) + tuple(comps))

# Final feature vector from user and business metadata
def extract_features(user_id, business_id, user_dict, business_dict, default_values):
    b = business_dict.get(business_id, default_values['business'])
    bstar, brev, lat, lon, pr, cc, appt, resv, tbl, whl = b
    brev_log = math.log1p(brev)

    u = user_dict.get(user_id, default_values['user'])
    urev, ufri, uuse, ufny, ucool, ufans, uelite, ustar, *comps = u
    urev_log = math.log1p(urev)

    return [bstar, brev_log, lat, lon, pr, cc, appt, resv, tbl, whl,
            urev_log, ufri, uuse, ufny, ucool, ufans, uelite, ustar] + comps

# ------------------- Main Pipeline -------------------
if __name__ == "__main__":
    # Initialize Spark
    conf = SparkConf().setAppName("HybridCFModel").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # File paths and config
    train_file = "yelp_train.csv"
    test_file = "yelp_val.csv"
    user_file = "user.json"
    business_file = "business.json"
    output_file = "output.csv"
    alpha = 0.3  # CF/XGB blend ratio

    # Load training and testing datasets
    train_rdd = sc.textFile(train_file).filter(lambda x: x != "user_id,business_id,stars")
    test_rdd = sc.textFile(test_file).filter(lambda x: x != "user_id,business_id,stars")

    train_data = train_rdd.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    test_data = test_rdd.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))

    # Create user and business rating dictionaries
    user_ratings = train_data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict)
    business_ratings = train_data.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict)

    # Compute mean ratings
    user_avg = user_ratings.mapValues(lambda x: sum(x.values()) / len(x)).collectAsMap()
    business_avg = business_ratings.mapValues(lambda x: sum(x.values()) / len(x)).collectAsMap()
    global_avg = train_data.map(lambda x: x[2]).mean()

    # Broadcast to all Spark workers
    user_ratings_dict = sc.broadcast(dict(user_ratings.collect())).value
    business_ratings_dict = sc.broadcast(dict(business_ratings.collect())).value
    user_avg_dict = sc.broadcast(user_avg).value
    business_avg_dict = sc.broadcast(business_avg).value

    # Load metadata from JSON files
    user_dict = sc.textFile(user_file).map(json.loads).map(parse_user).collectAsMap()
    business_dict = sc.textFile(business_file).map(json.loads).map(parse_business).collectAsMap()

    # Handle missing values with default profiles
    def avg_from_dict(d, idx):
        vals = [v[idx] for v in d.values()]
        return sum(vals) / len(vals) if vals else 0

    default_business = (
        avg_from_dict(business_dict, 0),
        avg_from_dict(business_dict, 1),
        0, 0, 2, 1, 0, 0, 0, 0
    )
    default_user = (
        avg_from_dict(user_dict, 0),
        5, 2, 0, 0, 0, 0,
        avg_from_dict(user_dict, 7)
    ) + tuple([0]*9)
    default_vals = {"business": default_business, "user": default_user}

    # Feature extraction and training set
    train_features = train_data.map(lambda x: (
        extract_features(x[0], x[1], user_dict, business_dict, default_vals), x[2]))
    X_train = train_features.map(lambda x: x[0]).collect()
    y_train = train_features.map(lambda x: x[1]).collect()

    # Prepare test features
    test_pairs = test_data.collect()
    X_test = [extract_features(u, b, user_dict, business_dict, default_vals) for u, b in test_pairs]

    # Train XGBoost model with bagging (5 random seeds)
    seeds = [3, 11, 29, 42, 77]
    bag_pred = [0.0] * len(X_test)
    for sd in seeds:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=5,
            learning_rate=0.1,
            n_estimators=1200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            seed=sd,
            verbosity=0,
            tree_method='hist'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        bag_pred = [a + b for a, b in zip(bag_pred, preds)]
    bag_pred = [p / len(seeds) for p in bag_pred]

    # Combine predictions from CF and XGBoost using weighted average
    final_preds = []
    for i, (user, business) in enumerate(test_pairs):
        cf = predict_cf(user, business, user_ratings_dict, business_ratings_dict,
                        user_avg_dict, business_avg_dict, global_avg)
        mb = bag_pred[i]
        hybrid = alpha * cf + (1 - alpha) * mb
        final_preds.append((user, business, hybrid))

    # Write predictions to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        writer.writerows(final_preds)

    sc.stop()
