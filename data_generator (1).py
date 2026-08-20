"""
data_generator.py
------------------
Generates a synthetic dataset of disputed (charged-back) transactions for
an Indian payments context. Each row is a "case" that has landed on the
RiskOps desk because the customer raised a chargeback / dispute.

The ground-truth label `is_fraud` tells us whether the dispute is a
GENUINE fraud case (someone else's card was used, i.e. the merchant
should win / decline the dispute) or a FRIENDLY / ILLEGITIMATE dispute
(the customer did make the purchase, i.e. merchant should contest and win,
customer should be treated as legitimate).

We deliberately bake in realistic, *imperfect* correlations between the
signals and the label so that the agents cannot get a trivial 100% score.
This lets the precision/recall/F1 numbers we report actually mean something.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_dataset(n_cases: int = 1200, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Base fraud rate for disputed transactions is much higher than the
    # overall payment base-rate, since these are already-flagged disputes.
    # Industry chargeback fraud studies commonly cite 30-45% of disputes as
    # "friendly fraud" style abuse vs genuine unauthorized-use fraud; we use
    # a 38% genuine-fraud rate here.
    is_fraud = rng.binomial(1, 0.38, size=n_cases).astype(bool)

    merchant_categories = np.array(
        ["electronics", "fashion", "food_delivery", "travel", "gaming", "utilities"]
    )
    merchant_category = rng.choice(merchant_categories, size=n_cases)

    # ---- Feature generation, correlated with is_fraud but noisy ----

    # 1. Transaction amount (INR). Fraud tends to skew towards higher-value,
    #    easily-resellable goods (electronics/gaming), with noise.
    base_amount = rng.lognormal(mean=8.0, sigma=0.9, size=n_cases)  # ~ hundreds to tens of thousands INR
    amount = base_amount * np.where(is_fraud, rng.uniform(1.1, 1.6, n_cases), rng.uniform(0.7, 1.1, n_cases))
    amount = np.round(amount, 2)

    # 2. Customer account age (days). Fraudsters often use newer / recently
    #    compromised accounts.
    account_age_days = np.where(
        is_fraud,
        rng.gamma(shape=2.0, scale=25, size=n_cases),      # skewed young
        rng.gamma(shape=6.0, scale=80, size=n_cases),      # skewed older
    )
    account_age_days = np.clip(account_age_days, 1, 3650).round(0)

    # 3. Merchant-side transaction velocity in the hour around this txn
    #    (how many txns the merchant processed in that hour) - proxy for
    #    a card-testing / fraud-ring spike hitting this merchant.
    merchant_velocity_1h = np.where(
        is_fraud,
        rng.poisson(lam=9, size=n_cases),
        rng.poisson(lam=3, size=n_cases),
    )

    # 4. Geo mismatch: distance (km) between billing address geolocation
    #    and the IP-derived geolocation at time of transaction.
    geo_mismatch_km = np.where(
        is_fraud,
        rng.gamma(shape=2.2, scale=350, size=n_cases),
        rng.gamma(shape=1.2, scale=40, size=n_cases),
    )
    geo_mismatch_km = np.clip(geo_mismatch_km, 0, 5000).round(1)

    # 5. New device flag - was this the first time this device fingerprint
    #    was seen on the account.
    new_device_prob = np.where(is_fraud, 0.78, 0.18)
    new_device = rng.binomial(1, new_device_prob).astype(bool)

    # 6. IP reputation risk score (0=clean, 1=known-bad / proxy-VPN / datacenter)
    ip_risk_score = np.where(
        is_fraud,
        np.clip(rng.beta(4, 2.2, size=n_cases), 0, 1),
        np.clip(rng.beta(1.5, 6, size=n_cases), 0, 1),
    )
    ip_risk_score = ip_risk_score.round(3)

    # 7. Prior chargebacks filed by this customer/card in last 180 days -
    #    a HIGH count is actually a signal of friendly-fraud / serial-disputer
    #    behaviour (i.e. NOT genuine fraud), so this one is INVERSELY
    #    correlated with is_fraud.
    prior_chargebacks_180d = np.where(
        is_fraud,
        rng.poisson(lam=0.15, size=n_cases),
        rng.poisson(lam=0.9, size=n_cases),
    )

    # 8. Time between login and checkout (minutes) - very fast sessions can
    #    indicate scripted/automated fraud; genuine shoppers browse longer.
    session_to_checkout_min = np.where(
        is_fraud,
        rng.exponential(scale=2.0, size=n_cases),
        rng.exponential(scale=9.0, size=n_cases),
    )
    session_to_checkout_min = np.clip(session_to_checkout_min, 0.1, 120).round(2)

    # 9. Card-present / saved-card reuse flag - genuine repeat customers are
    #    more likely to be using a long-saved card.
    saved_card_age_days = np.where(
        is_fraud,
        rng.exponential(scale=10, size=n_cases),
        rng.exponential(scale=180, size=n_cases),
    )
    saved_card_age_days = np.clip(saved_card_age_days, 0, 2000).round(0)

    # ---- Realism injection ----
    # No fraud signal set is perfect in the real world. We deliberately
    # corrupt a slice of cases so the classifier cannot achieve trivial
    # separation:
    #   (a) "sophisticated fraud" (~16% of fraud cases) - the fraudster used
    #       a residential proxy near the billing address, an aged/stolen
    #       saved-card token, and browsed like a normal shopper. These look
    #       clean on every signal even though they are genuine fraud.
    #   (b) "unlucky genuine customers" (~10% of legit disputes) - a real
    #       customer travelling abroad, on a new phone, at a merchant
    #       having a genuine traffic spike. These trip several flags even
    #       though the dispute is legitimate (e.g. they simply forgot
    #       about a subscription renewal).
    sophisticated_fraud = is_fraud & (rng.random(n_cases) < 0.16)
    unlucky_genuine = (~is_fraud) & (rng.random(n_cases) < 0.10)

    merchant_velocity_1h = np.where(sophisticated_fraud, rng.poisson(lam=2, size=n_cases), merchant_velocity_1h)
    merchant_velocity_1h = np.where(unlucky_genuine, rng.poisson(lam=8, size=n_cases), merchant_velocity_1h)

    geo_mismatch_km = np.where(sophisticated_fraud, rng.gamma(2, 30, size=n_cases), geo_mismatch_km)
    geo_mismatch_km = np.where(unlucky_genuine, rng.gamma(2.5, 300, size=n_cases), geo_mismatch_km)  # e.g. genuinely travelling

    new_device = np.where(sophisticated_fraud, rng.binomial(1, 0.15, n_cases).astype(bool), new_device)
    new_device = np.where(unlucky_genuine, rng.binomial(1, 0.75, n_cases).astype(bool), new_device)

    ip_risk_score = np.where(sophisticated_fraud, np.clip(rng.beta(1.5, 6, n_cases), 0, 1).round(3), ip_risk_score)
    ip_risk_score = np.where(unlucky_genuine, np.clip(rng.beta(3, 3, n_cases), 0, 1).round(3), ip_risk_score)

    session_to_checkout_min = np.where(sophisticated_fraud, rng.exponential(9, n_cases), session_to_checkout_min)
    session_to_checkout_min = np.where(unlucky_genuine, rng.exponential(1.5, n_cases), session_to_checkout_min)
    session_to_checkout_min = np.clip(session_to_checkout_min, 0.1, 120).round(2)

    saved_card_age_days = np.where(sophisticated_fraud, rng.exponential(220, n_cases), saved_card_age_days)
    saved_card_age_days = np.where(unlucky_genuine, rng.exponential(8, n_cases), saved_card_age_days)
    saved_card_age_days = np.clip(saved_card_age_days, 0, 2000).round(0)

    case_id = [f"CB-{100000+i}" for i in range(n_cases)]

    df = pd.DataFrame(
        {
            "case_id": case_id,
            "merchant_category": merchant_category,
            "amount_inr": amount,
            "customer_account_age_days": account_age_days,
            "merchant_velocity_1h": merchant_velocity_1h,
            "geo_mismatch_km": geo_mismatch_km,
            "new_device": new_device,
            "ip_risk_score": ip_risk_score,
            "prior_chargebacks_180d": prior_chargebacks_180d,
            "session_to_checkout_min": session_to_checkout_min,
            "saved_card_age_days": saved_card_age_days,
            "is_fraud": is_fraud,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("disputes_dataset.csv", index=False)
    print(df.head(10).to_string())
    print(f"\nTotal cases: {len(df)}  |  Fraud rate: {df['is_fraud'].mean():.1%}")
