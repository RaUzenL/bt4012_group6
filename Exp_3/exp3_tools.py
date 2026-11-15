"""Experiment 3: the six evidence tools the investigation agent can call.

Each tool takes one applicant ID and returns a small, flat summary (a dict) of ONE data source.
The tools do the counting and the arithmetic in pandas. The LLM only reads the summaries.

Design rules (from the Week 9 lecture):
* Read-only. No tool can change anything.
* Analysed summary, not raw rows. Every value is a single number, word or true/false.
* Least privilege. Gender, marital status and age are never loaded, so no tool can return them.
  TARGET (the label) is never loaded either.
* Missing history is reported as missing ("has_record": false). It is not a clean record.
* A bad ID returns an error dict. The tools never raise, so one bad call cannot stop an investigation.

Field names are unique across the six tools, so a citation can be traced back to one tool.
The docstring of each tool is the description the LLM reads when it decides which tool to call.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Places where the shared folder (made by data_cleaning.ipynb) may be, depending on whose laptop this is.
SHARED_CANDIDATES = [
    HERE.parent / "outputs" / "outputs",
    HERE.parent / "hongwei_outputs" / "outputs",
    HERE.parent / "outputs",
]

# Only these columns are read. Protected attributes and TARGET are not in the lists.
COLUMNS = {
    "application": [
        "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_HOUSING_TYPE",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_RATING_CLIENT",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
        "DEF_30_CNT_SOCIAL_CIRCLE", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
    "bureau": [
        "SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    ],
    "bureau_balance": ["SK_ID_CURR", "MONTHS_BALANCE", "STATUS_DPD"],
    "previous_application": [
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "IS_DUPLICATE_SUBMISSION", "NAME_YIELD_GROUP",
    ],
    "installments": [
        "SK_ID_CURR", "SK_ID_PREV", "DAYS_INSTALMENT", "AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_ENTRY_PAYMENT",
    ],
    "credit_card": [
        "SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "CNT_DRAWINGS_ATM_CURRENT", "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_TOTAL_CURRENT", "SK_DPD",
    ],
    "pos_cash": ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS", "SK_DPD"],
}

# Worst monthly status at another lender, as words the report can quote.
BUREAU_STATUS_WORDS = {
    0: "never late",
    1: "1-30 days late",
    2: "31-60 days late",
    3: "61-90 days late",
    4: "91-120 days late",
    5: "over 120 days late or written off",
}

DATA = {}    # table name -> DataFrame indexed by SK_ID_CURR (filled by load_data)
TRAIN = {}   # sorted values of TRAIN applicants, used for the percentile fields


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def find_shared_folder():
    """Return the shared folder that holds clean_data/ and exp3/."""
    for folder in SHARED_CANDIDATES:
        if (folder / "clean_data" / "splits.csv").exists():
            return folder
    raise FileNotFoundError("Shared folder not found. Tried: {0}".format([str(f) for f in SHARED_CANDIDATES]))


def load_data(shared_folder=None):
    """Read the cleaned tables once and index them by applicant ID."""
    if shared_folder is None:
        shared_folder = find_shared_folder()
    clean_folder = Path(shared_folder) / "clean_data"

    for table_name, columns in COLUMNS.items():
        table = pd.read_parquet(clean_folder / (table_name + ".parquet"), columns=columns)
        DATA[table_name] = table.set_index("SK_ID_CURR").sort_index()

    # Percentiles compare an applicant with TRAIN applicants only (shared rule: fit on train).
    splits = pd.read_csv(clean_folder / "splits.csv", usecols=["SK_ID_CURR", "split"])
    train_ids = splits.loc[splits["split"] == "train", "SK_ID_CURR"]
    application = DATA["application"]
    train_rows = application[application.index.isin(train_ids)]

    burden = train_rows["AMT_ANNUITY"] / train_rows["AMT_INCOME_TOTAL"]
    external = train_rows[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    TRAIN["annuity_to_income"] = np.sort(burden.dropna().to_numpy())
    TRAIN["external_score_mean"] = np.sort(external.dropna().to_numpy())
    return DATA


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def _number(value, digits=4):
    """Round a number for the summary. A missing value becomes None (null in JSON)."""
    if value is None or pd.isna(value):
        return None
    if digits == 0:
        return int(round(float(value)))
    return round(float(value), digits)


def _text(value):
    """A category as plain text, or None if it is missing."""
    if value is None or pd.isna(value):
        return None
    return str(value)


def _percentile(value, name):
    """Where a value sits among TRAIN applicants: 0 = lowest, 100 = highest."""
    if value is None or pd.isna(value):
        return None
    reference = TRAIN[name]
    position = np.searchsorted(reference, value, side="right")
    return int(round(100.0 * position / len(reference)))


def _check_applicant(applicant_id):
    """Return (ID as int, None) for a known applicant, or (None, error dict)."""
    if not DATA:
        load_data()
    try:
        checked_id = int(applicant_id)
    except (TypeError, ValueError):
        error = {"error": "bad_argument",
                 "message": "applicant_id must be a whole number such as 100054. Retry with the ID from the case."}
        return None, error
    if checked_id not in DATA["application"].index:
        error = {"error": "not_found",
                 "message": "No applicant {0}. Re-check the ID in the case. This is not evidence about the applicant.".format(checked_id)}
        return None, error
    return checked_id, None


def _rows(table_name, applicant_id):
    """All rows of one history table for one applicant (an empty table if there are none)."""
    table = DATA[table_name]
    if applicant_id not in table.index:
        return table.iloc[0:0]
    return table.loc[[applicant_id]]


def _no_record(applicant_id, what):
    """The answer when an applicant has no rows in a history table."""
    note = "No {0} on file. This is missing history, not a clean history.".format(what)
    return {"applicant_id": applicant_id, "has_record": False, "note": note, "truncated": False}


# ----------------------------------------------------------------------------------------------
# Tool 1: the application form
# ----------------------------------------------------------------------------------------------
def get_applicant_profile(applicant_id: int) -> dict:
    """Return the current loan application: loan size, income, repayment burden, employment,
    education, external credit scores, recent ID/phone changes and credit-bureau enquiries.
    Use it once for every applicant. It does NOT include any repayment or credit history
    (use the other five tools) and it withholds gender, marital status and age.
    Ratios are fractions. A '_percentile' field is the position among past applicants
    (0 = lowest, 100 = highest). external_score_mean runs from 0 to 1 and higher is safer.
    credit_to_goods_price above 1 means the loan is larger than the goods it pays for.
    years_employed is null when no current job is on file (mostly pensioners); that is
    missing information, not a short employment."""
    applicant_id, error = _check_applicant(applicant_id)
    if error:
        return error
    row = DATA["application"].loc[applicant_id]

    income = row["AMT_INCOME_TOTAL"]
    annuity_to_income = row["AMT_ANNUITY"] / income
    credit_to_income = row["AMT_CREDIT"] / income
    if row["AMT_GOODS_PRICE"] > 0:
        credit_to_goods_price = row["AMT_CREDIT"] / row["AMT_GOODS_PRICE"]
    else:
        credit_to_goods_price = None                 # no goods price on file (for example a revolving loan)

    external_scores = row[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype("float64")
    external_score_mean = external_scores.mean()     # skips missing scores; NaN if all three are missing

    address_mismatches = row["REG_CITY_NOT_LIVE_CITY"] + row["REG_CITY_NOT_WORK_CITY"] + row["LIVE_CITY_NOT_WORK_CITY"]

    # DAYS_* columns count backwards from the application day, so they are negative.
    return {
        "applicant_id": applicant_id,
        "contract_type": _text(row["NAME_CONTRACT_TYPE"]),
        "credit_amount": _number(row["AMT_CREDIT"], 0),
        "goods_price": _number(row["AMT_GOODS_PRICE"], 0),
        "annuity": _number(row["AMT_ANNUITY"], 0),
        "income_total": _number(income, 0),
        "annuity_to_income": _number(annuity_to_income),
        "annuity_to_income_percentile": _percentile(annuity_to_income, "annuity_to_income"),
        "credit_to_income": _number(credit_to_income, 2),
        "credit_to_goods_price": _number(credit_to_goods_price, 2),
        "income_type": _text(row["NAME_INCOME_TYPE"]),
        "education": _text(row["NAME_EDUCATION_TYPE"]),
        "occupation": _text(row["OCCUPATION_TYPE"]),
        "organization_type": _text(row["ORGANIZATION_TYPE"]),
        "housing_type": _text(row["NAME_HOUSING_TYPE"]),
        "owns_car": _text(row["FLAG_OWN_CAR"]) == "Y",
        "owns_realty": _text(row["FLAG_OWN_REALTY"]) == "Y",
        "years_employed": _number(-row["DAYS_EMPLOYED"] / 365.0, 3),
        "years_since_id_change": _number(-row["DAYS_ID_PUBLISH"] / 365.0, 3),
        "years_since_registration_change": _number(-row["DAYS_REGISTRATION"] / 365.0, 3),
        "days_since_phone_change": _number(-row["DAYS_LAST_PHONE_CHANGE"], 0),
        "external_score_mean": _number(external_score_mean),
        "external_score_percentile": _percentile(external_score_mean, "external_score_mean"),
        "n_external_scores": int(external_scores.notna().sum()),
        "bureau_enquiries_last_year": _number(row["AMT_REQ_CREDIT_BUREAU_YEAR"], 0),
        "region_rating": _number(row["REGION_RATING_CLIENT"], 0),
        "address_mismatch_count": _number(address_mismatches, 0),
        "social_circle_defaults": _number(row["DEF_30_CNT_SOCIAL_CIRCLE"], 0),
        "truncated": False,
    }


# ----------------------------------------------------------------------------------------------
# Tool 2: loans at other lenders (credit bureau)
# ----------------------------------------------------------------------------------------------
def get_credit_bureau_history(applicant_id: int) -> dict:
    """Return the applicant's loans at OTHER lenders as reported by the credit bureau:
    number of loans, how recent and how old they are, active debt against active credit,
    amount overdue now, and the worst monthly lateness on record.
    Use it to check debt burden, recent borrowing and delinquency outside Home Credit.
    It does NOT cover Home Credit's own loans (see the previous-application, instalment,
    credit-card and POS/cash tools). bureau_worst_monthly_status is null when the bureau
    holds no month-by-month records, which is missing evidence, not a clean record."""
    applicant_id, error = _check_applicant(applicant_id)
    if error:
        return error
    loans = _rows("bureau", applicant_id)
    if len(loans) == 0:
        return _no_record(applicant_id, "credit bureau loans at other lenders")

    status = loans["CREDIT_ACTIVE"].astype(str)
    active = loans[status == "Active"]
    active_credit = active["AMT_CREDIT_SUM"].astype("float64").sum()
    active_debt = active["AMT_CREDIT_SUM_DEBT"].astype("float64").sum()
    if active_credit > 0:
        active_debt_to_credit = active_debt / active_credit
    else:
        active_debt_to_credit = None

    # Month-by-month status. Only some loans have it.
    months = _rows("bureau_balance", applicant_id)
    known_months = months["STATUS_DPD"].dropna()
    if len(known_months) > 0:
        worst_status = BUREAU_STATUS_WORDS[int(known_months.max())]
        months_late = int((known_months >= 1).sum())
        in_last_year = months["MONTHS_BALANCE"] >= -12          # month 0 is the application month
        months_late_last_year = int((months.loc[in_last_year, "STATUS_DPD"] >= 1).sum())
    else:
        worst_status = None
        months_late = None
        months_late_last_year = None

    return {
        "applicant_id": applicant_id,
        "has_record": True,
        "bureau_n_loans": len(loans),
        "bureau_n_active_loans": len(active),
        "bureau_n_closed_loans": int((status == "Closed").sum()),
        "bureau_active_credit": _number(active_credit, 0),
        "bureau_active_debt": _number(active_debt, 0),
        "bureau_active_debt_to_credit": _number(active_debt_to_credit),
        "bureau_overdue_amount": _number(loans["AMT_CREDIT_SUM_OVERDUE"].astype("float64").sum(), 0),
        "bureau_max_days_overdue_now": _number(loans["CREDIT_DAY_OVERDUE"].max(), 0),
        "bureau_max_overdue_amount_ever": _number(loans["AMT_CREDIT_MAX_OVERDUE"].max(), 0),
        "bureau_n_loans_opened_last_year": int((loans["DAYS_CREDIT"] >= -365).sum()),
        "bureau_days_since_last_loan_opened": _number(-loans["DAYS_CREDIT"].max(), 0),
        "bureau_mean_loan_age_years": _number(-loans["DAYS_CREDIT"].mean() / 365.0, 2),
        "bureau_n_prolonged": int(loans["CNT_CREDIT_PROLONG"].sum()),
        "bureau_months_reported": len(known_months),
        "bureau_months_late": months_late,
        "bureau_months_late_last_year": months_late_last_year,
        "bureau_worst_monthly_status": worst_status,
        "truncated": False,
    }


# ----------------------------------------------------------------------------------------------
# Tool 3: earlier applications to Home Credit
# ----------------------------------------------------------------------------------------------
def get_previous_applications(applicant_id: int) -> dict:
    """Return the applicant's EARLIER applications to Home Credit: how many, how many were
    approved or refused, the outcome of the most recent one, how many were made in the
    last year, and how Home Credit priced the approved loans.
    Use it to check rejection history and application frequency.
    prev_high_interest_share is the share of approved loans that Home Credit put in its
    high interest-rate group, which is how it prices riskier customers.
    It does NOT show how approved loans were repaid (see get_installment_history).
    Repeat submissions of the same application are counted separately in
    prev_n_duplicate_submissions and left out of every other field. prev_last_status is
    'Refused' if any application decided on the most recent decision day was refused."""
    applicant_id, error = _check_applicant(applicant_id)
    if error:
        return error
    all_rows = _rows("previous_application", applicant_id)
    applications = all_rows[all_rows["IS_DUPLICATE_SUBMISSION"] == 0]
    if len(applications) == 0:
        return _no_record(applicant_id, "previous Home Credit applications")

    status = applications["NAME_CONTRACT_STATUS"].astype(str)
    n_refused = int((status == "Refused").sum())

    approved = applications[status == "Approved"]
    if len(approved) > 0:
        high_interest_share = (approved["NAME_YIELD_GROUP"].astype(str) == "high").mean()
    else:
        high_interest_share = None

    last_day = applications["DAYS_DECISION"].max()
    status_on_last_day = status[applications["DAYS_DECISION"] == last_day]
    if (status_on_last_day == "Refused").any():
        last_status = "Refused"
    elif (status_on_last_day == "Approved").any():
        last_status = "Approved"
    else:
        last_status = status_on_last_day.iloc[0]

    return {
        "applicant_id": applicant_id,
        "has_record": True,
        "prev_n_applications": len(applications),
        "prev_n_approved": int((status == "Approved").sum()),
        "prev_n_refused": n_refused,
        "prev_n_canceled_or_unused": int(status.isin(["Canceled", "Unused offer"]).sum()),
        "prev_refusal_rate": _number(n_refused / len(applications)),
        "prev_last_status": last_status,
        "prev_days_since_last_decision": _number(-last_day, 0),
        "prev_n_applications_last_year": int((applications["DAYS_DECISION"] >= -365).sum()),
        "prev_high_interest_share": _number(high_interest_share, 2),
        "prev_n_duplicate_submissions": int((all_rows["IS_DUPLICATE_SUBMISSION"] == 1).sum()),
        "truncated": False,
    }


# ----------------------------------------------------------------------------------------------
# Tool 4: instalment payments on earlier Home Credit loans
# ----------------------------------------------------------------------------------------------
def get_installment_history(applicant_id: int) -> dict:
    """Return how the applicant repaid instalments on earlier Home Credit loans: share paid
    late, worst lateness in days, share underpaid, and whether lateness got worse in the last
    year. Use it to check repayment behaviour. It covers Home Credit loans only (other
    lenders are in get_credit_bureau_history) and it does not cover credit cards.
    Rates are fractions of all instalments. inst_late_rate_change is the last-year late rate
    minus the overall late rate, so a positive value means repayment is getting worse.
    inst_mean_days_early is how many days before the due date the applicant pays on average."""
    applicant_id, error = _check_applicant(applicant_id)
    if error:
        return error
    rows = _rows("installments", applicant_id)
    if len(rows) == 0:
        return _no_record(applicant_id, "instalment payments on earlier Home Credit loans")

    days_late = rows["DAYS_ENTRY_PAYMENT"] - rows["DAYS_INSTALMENT"]    # missing if nothing was paid
    amount_due = rows["AMT_INSTALMENT"].astype("float64")
    amount_paid = rows["AMT_PAYMENT"].fillna(0).astype("float64")
    is_late = days_late > 0
    is_underpaid = (amount_due - amount_paid) > 1.0
    due_last_year = rows["DAYS_INSTALMENT"] >= -365

    late_rate = is_late.mean()
    if due_last_year.any():
        late_rate_last_year = is_late[due_last_year].mean()
        late_rate_change = late_rate_last_year - late_rate
    else:
        late_rate_last_year = None
        late_rate_change = None

    if amount_due.sum() > 0:
        paid_to_due = amount_paid.sum() / amount_due.sum()
    else:
        paid_to_due = None

    return {
        "applicant_id": applicant_id,
        "has_record": True,
        "inst_n_loans": int(rows["SK_ID_PREV"].nunique()),
        "inst_n_installments": len(rows),
        "inst_n_unpaid": int(rows["AMT_PAYMENT"].isna().sum()),
        "inst_late_rate": _number(late_rate),
        "inst_max_days_late": _number(days_late.clip(lower=0).max(), 0),
        "inst_mean_days_late": _number(days_late.clip(lower=0).mean(), 1),
        "inst_mean_days_early": _number((-days_late).clip(lower=0).mean(), 1),
        "inst_underpaid_rate": _number(is_underpaid.mean()),
        "inst_paid_to_due_ratio": _number(paid_to_due),
        "inst_n_installments_last_year": int(due_last_year.sum()),
        "inst_late_rate_last_year": _number(late_rate_last_year),
        "inst_late_rate_change": _number(late_rate_change),
        "truncated": False,
    }


# ----------------------------------------------------------------------------------------------
# Tool 5: Home Credit credit cards
# ----------------------------------------------------------------------------------------------
def get_credit_card_history(applicant_id: int) -> dict:
    """Return the monthly history of the applicant's Home Credit credit cards: latest balance
    against the limit, months paid below the minimum payment, share of card spending taken as
    cash, how often cash is withdrawn, and days past due.
    Use it to check card utilisation and payment discipline.
    Most applicants have no Home Credit card, so has_record=false is common and is missing
    history, not a clean record. Cards at other lenders are in get_credit_bureau_history.
    Utilisation and the cash share are fractions (0.9 means 90%)."""
    applicant_id, error = _check_applicant(applicant_id)
    if error:
        return error
    rows = _rows("credit_card", applicant_id)
    if len(rows) == 0:
        return _no_record(applicant_id, "Home Credit credit card")

    # One row per month, adding up all of the applicant's cards.
    monthly = rows.groupby("MONTHS_BALANCE")[["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]].sum()
    monthly = monthly.astype("float64")
    monthly_utilisation = monthly["AMT_BALANCE"] / monthly["AMT_CREDIT_LIMIT_ACTUAL"].where(monthly["AMT_CREDIT_LIMIT_ACTUAL"] > 0)
    latest_month = monthly.index.max()

    minimum_due = rows["AMT_INST_MIN_REGULARITY"]
    paid = rows["AMT_PAYMENT_TOTAL_CURRENT"].fillna(0)
    below_minimum = (minimum_due > 0) & (paid < minimum_due)

    cash_drawn = rows["AMT_DRAWINGS_ATM_CURRENT"].astype("float64").sum()
    total_drawn = rows["AMT_DRAWINGS_CURRENT"].astype("float64").sum()
    if total_drawn != 0:
        cash_share = cash_drawn / total_drawn
    else:
        cash_share = None
    atm_withdrawals_per_month = rows["CNT_DRAWINGS_ATM_CURRENT"].astype("float64").mean()   # NaN if never recorded

    return {
        "applicant_id": applicant_id,
        "has_record": True,
        "card_n_cards": int(rows["SK_ID_PREV"].nunique()),
        "card_months_of_history": int(rows["MONTHS_BALANCE"].nunique()),
        "card_latest_balance": _number(monthly.loc[latest_month, "AMT_BALANCE"], 0),
        "card_latest_limit": _number(monthly.loc[latest_month, "AMT_CREDIT_LIMIT_ACTUAL"], 0),
        "card_latest_utilisation": _number(monthly_utilisation.loc[latest_month]),
        "card_mean_utilisation": _number(monthly_utilisation.mean()),
        "card_months_below_min_payment": int(below_minimum.sum()),
        "card_cash_share_of_drawings": _number(cash_share),
        "card_atm_withdrawals_per_month": _number(atm_withdrawals_per_month, 2),
        "card_max_days_past_due": _number(rows["SK_DPD"].max(), 0),
        "card_months_past_due": int((rows["SK_DPD"] > 0).sum()),
        "truncated": False,
    }


# ----------------------------------------------------------------------------------------------
# Tool 6: Home Credit point-of-sale and cash loans
# ----------------------------------------------------------------------------------------------
def get_pos_cash_history(applicant_id: int) -> dict:
    """Return the monthly history of the applicant's Home Credit point-of-sale and cash loans:
    worst days past due, number of months past due, and whether any contract was sent to debt
    collection ('Demand' status). Use it to check delinquency on these loans.
    It does NOT include payment amounts (see get_installment_history) or credit cards.
    has_record=false means the applicant never had such a loan, which is missing history."""
    applicant_id, error = _check_applicant(applicant_id)
    if error:
        return error
    rows = _rows("pos_cash", applicant_id)
    if len(rows) == 0:
        return _no_record(applicant_id, "Home Credit POS or cash loan")

    status = rows["NAME_CONTRACT_STATUS"].astype(str)
    latest_rows = rows.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    latest_status = latest_rows["NAME_CONTRACT_STATUS"].astype(str)

    return {
        "applicant_id": applicant_id,
        "has_record": True,
        "pos_n_contracts": int(rows["SK_ID_PREV"].nunique()),
        "pos_n_active_contracts": int((latest_status == "Active").sum()),
        "pos_months_of_history": int(rows["MONTHS_BALANCE"].nunique()),
        "pos_max_days_past_due": _number(rows["SK_DPD"].max(), 0),
        "pos_months_past_due": int((rows["SK_DPD"] > 0).sum()),
        "pos_months_in_debt_demand": int((status == "Demand").sum()),
        "truncated": False,
    }


# The agent's whole toolset. There is no write tool, so the agent cannot approve or decline a loan.
TOOLS = [
    get_applicant_profile,
    get_credit_bureau_history,
    get_previous_applications,
    get_installment_history,
    get_credit_card_history,
    get_pos_cash_history,
]
TOOL_NAMES = [tool.__name__ for tool in TOOLS]
