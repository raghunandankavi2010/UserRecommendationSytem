"""python -m recsys.cli --user 7 [--method hybrid] [--n 5] [--no-signals] [--rebuild]"""
import argparse

import pandas as pd

from . import db
from .models import DEFAULT_WEIGHTS, METHODS, Recommender


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend people to connect with, by shared interests.")
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--method", choices=METHODS, default="hybrid")
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid weight on TF-IDF")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-signals", action="store_true", help="interests only, no mutual/place/recency re-rank")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--rebuild", action="store_true", help="recreate the synthetic database")
    args = ap.parse_args()

    if args.rebuild or not db.Path(args.db).exists():
        db.build_db(args.db)
    conn = db.connect(args.db)
    profiles = db.load_profiles(conn)
    if args.user not in set(profiles["user_id"]):
        ap.error(f"user {args.user} not found or has no interests")

    weights = None if args.no_signals else DEFAULT_WEIGHTS
    model = Recommender(args.method, alpha=args.alpha, weights=weights).fit(
        profiles, db.load_connections(conn), db.load_signals(conn))
    me = profiles.set_index("user_id").loc[args.user]
    print(f"{me['name']} (#{args.user}) likes: {', '.join(me['interests'])}\n")
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(model.recommend(args.user, args.n, db.candidate_ids(conn, args.user)).to_string(index=False))


if __name__ == "__main__":
    main()
