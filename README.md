# Customer Feedback Analysis and Automated Response

## Final submission

This submission contains two files:

1. `final_electronics_customer_feedback_capstone.ipynb`
2. `README.md`

The notebook is the complete implementation and demonstration. The README explains the business problem, dataset choice, workflow, decisions, evaluation method, Gemini stage, security, limitations, and expected results.

---

## 1. Project objective

An online retail support team can receive a large number of customer reviews. Manually reading every review to find urgent complaints and then writing an initial response takes time.

This project builds a transparent Python workflow that:

- loads customer-review data;
- profiles and audits the raw data;
- preprocesses the review text;
- performs focused exploratory analysis;
- identifies Critical reviews using a rule;
- finds common complaint areas using a human-defined dictionary;
- selects the three most severe and detailed Critical reviews;
- uses Gemini to generate short, empathetic response-email drafts;
- allows a reviewer to select one of the three cases and provide an additional comment;
- asks Gemini to generate a revised draft using both the selected review and the additional comment.

The project does **not** train a machine-learning model and does **not** automatically send email.

---

## 2. Why this dataset was selected

The project uses the public Mendeley Data dataset **“sentiment in customer reviews”**, version 1, DOI `10.17632/2zfbcy96vp.1`.

Source:
https://data.mendeley.com/datasets/2zfbcy96vp/1

The dataset page states that the data contains reviews from 3,600 customers. The working CSV used in this project contains:

- 3,600 rows;
- 11 raw columns;
- 1–5 star ratings;
- product information;
- review title and review text;
- review date;
- verified-purchase information;
- product format;
- helpful-vote information.

### Why it fits this project

The dataset was selected because it contains the two pieces of information required by the assignment:

1. a customer review text;
2. a 1–5 star rating.

It is also small enough to process on a normal laptop without sampling. The dataset is directly related to online retail and consumer feedback, so the business problem is easy to explain.

---

## 3. Complete workflow

The notebook follows this order:

```text
Raw customer reviews
        ↓
Data loading
        ↓
Profiling: shape, info, describe
        ↓
Data audit
        ↓
Preprocessing
        ↓
Target-focused EDA
        ↓
EDA inference
        ↓
Critical rule: 1–2 stars
        ↓
Complaint dictionary + Counter
        ↓
Deterministic top-three selection
        ↓
Secure Gemini API-key input
        ↓
Live Gemini generation
        ↓
Three response drafts
        ↓
Reviewer selects 1 / 2 / 3
        ↓
Reviewer enters additional comment
        ↓
Gemini generates revised draft
        ↓
Human review before real use
```

The order matters. Gemini is deliberately placed after the analytical work so that the LLM does not decide which reviews are Critical.

---

# 4. Step 1 — Load the data

The notebook loads the original CSV with Pandas.

The raw dataset is not modified at this stage.

The notebook first checks common local paths. If the file is not found, it asks the evaluator to enter the path to:

```text
amazon_electronics_reviews.csv
```

This keeps the notebook independent of one person's computer path.

The automatic-download fallback uses the public Mendeley dataset archive. An internet connection is therefore useful for a two-file submission, but a local CSV path can also be supplied.

---

# 5. Step 2 — Data profiling

Before cleaning anything, the notebook shows:

- `shape`;
- `info()`;
- numeric `describe()`;
- `describe(include="all")`.

### Why?

Profiling tells us what the dataset actually contains before we make cleaning decisions.

The working dataset has:

- 3,600 rows;
- 11 raw columns;
- ratings from 1 to 5.

The rating distribution is:

| Rating | Reviews |
|---:|---:|
| 1 | 495 |
| 2 | 192 |
| 3 | 287 |
| 4 | 732 |
| 5 | 1,894 |

---

# 6. Step 3 — Data audit

The audit is performed before preprocessing.

The notebook checks:

- missing values;
- exact duplicate rows;
- invalid or missing ratings;
- missing review text;
- possible duplicate-review candidates.

Important observed findings:

- Missing review text: 25
- Missing product format: 904
- Missing helpful votes: 1,285
- Missing author: 1
- Exact duplicate rows: 0
- Invalid/missing ratings: 0

There are also possible duplicate-review candidates based on product + title + comment. They are flagged, not automatically deleted.

### Why not delete every row with a missing value?

Because missingness has different meanings.

For example, a missing helpful-vote field does not prove that the customer received zero helpful votes. Automatically converting missing values to zero would create information that was not present in the source.

---

# 7. Step 4 — Preprocessing

The notebook creates a processed DataFrame without overwriting the raw data.

### Main preprocessing actions

1. Remove the source index artifact `Unnamed: 0`.
2. Standardize important column names.
3. Convert rating to numeric.
4. Parse the review date.
5. Create `review_text_original`.
6. Create a separate `review_text_clean`.
7. Create a missing-review flag.
8. Calculate review length in words.
9. Create a stable review ID.
10. Create `critical_flag`.

### Original text vs cleaned text

This distinction is important.

```text
review_text_original
        ↓
kept for audit + Gemini
```

while:

```text
review_text_clean
        ↓
used for rule-based keyword matching
```

The original customer wording is therefore not lost.

### Text cleaning

Only the separate cleaned copy is changed:

- lowercase;
- remove URLs;
- remove special characters;
- normalize spaces.

The cleaning is deliberately simple because the assignment asks for rule-based analysis, not advanced NLP.

---

# 8. Step 5 — Target-focused EDA

For this project, `rating` is treated as the practical **target variable for triage**.

This does **not** mean a machine-learning model is trained.

The notebook uses a small EDA focused on the rating:

- rating distribution;
- Critical 1–2 star distribution.

### Why rating?

The assignment defines urgent/Critical feedback using low ratings. Therefore, rating is the business variable that drives the triage decision.

### EDA inference

The dataset is strongly concentrated toward high ratings, with 5-star reviews being the largest group.

However:

```text
495 one-star
+ 192 two-star
= 687 Critical reviews
```

Therefore:

```text
687 / 3600 × 100 = 19.08%
```

About 19% of the reviews enter the Critical queue.

The EDA describes the observed data. It does not claim that a particular product caused a rating.

---

# 9. Step 6 — Rule-based Critical filtering

The assignment requires rule-based filtering and specifically gives the 1–2 star idea.

The notebook uses:

```python
processed["rating"].isin([1, 2])
```

Therefore:

```text
Critical = 1-star OR 2-star
```

Result:

**687 Critical reviews out of 3,600.**

No machine-learning classifier is used.

### Why not ML?

Because the assignment asks for rule-based filtering.

A rule is:

> a human-defined condition that the program applies consistently.

For example:

```text
IF rating is 1 or 2
THEN Critical
```

This is transparent and easy to audit.

---

# 10. Step 7 — Complaint dictionary and Counter

After Critical reviews are identified, Python searches their cleaned text for electronics-related complaint phrases.

The notebook contains 11 complaint categories:

- battery
- charging
- heating
- camera
- display
- audio
- connectivity
- software
- quality or damage
- delivery or packaging
- value or expectation

The dictionary is a human-defined rulebook.

### How it works

Example:

```text
Customer review:
"The battery drains quickly and charging is very slow."

Python detects:
battery
charging
```

`Counter` then summarizes how frequently terms/categories occur across Critical reviews.

### Why can category counts overlap?

Because one customer can complain about several issues in the same review.

Therefore, category counts do not have to add up to 687.

### Limitation

Keyword matching cannot fully understand:

- synonyms not in the dictionary;
- spelling variations;
- negation;
- context.

This limitation is explicitly acknowledged rather than presenting the rule-based method as perfect sentiment analysis.

---

# 11. Step 8 — Select the top three Critical reviews

Gemini does not choose the reviews.

Python selects exactly three using a deterministic priority rule:

1. lower rating first;
2. more complaint categories;
3. longer review;
4. stable review ID as the final tie-breaker.

This makes the selection reproducible.

Review length is only a supporting signal because a long review can provide more detail, but length alone does not mean greater urgency.

---

# 12. Step 9 — Gemini API key

The Gemini stage is live.

The notebook asks for the API key at runtime using:

```python
getpass.getpass(...)
```

The key is therefore hidden from normal notebook display.

### Important security clarification

The key is **not hashed**.

Hashing is not the correct mechanism for an API credential because the original credential is needed to authenticate the API request.

The correct security approach is:

```text
User enters key
      ↓
Input is not echoed
      ↓
Client is created
      ↓
Key is not printed
      ↓
Key is not written into the notebook
```

The notebook does not contain the user's real key.

Never place the key directly in a code cell or commit it to GitHub.

---

# 13. Step 10 — Live Gemini generation

After the API key is entered, Gemini receives the three selected Critical reviews.

For each selected review, the prompt tells Gemini to:

- write a clear subject;
- apologize naturally;
- address the actual complaint;
- keep the draft short;
- use the original review as the source of truth;
- translate internal complaint categories into natural customer language;
- avoid invented customer names;
- avoid invented order numbers;
- avoid invented refunds/replacements/discounts;
- avoid invented policies or tracking details;
- avoid claiming an action has already happened;
- ask the customer to contact the official support channel;
- produce a human-review draft.

### What Gemini does

Gemini is the **language-generation layer**.

### What Gemini does not do

Gemini does not:

- decide whether a review is Critical;
- select the top three;
- calculate complaint counts;
- approve a refund;
- send an email.

This separation is intentional:

```text
Python → analysis and decision
Gemini → response drafting
Human → final approval
```

---

# 14. The AFC warning and why it is disabled

The current `google-genai` SDK can display an Automatic Function Calling (AFC) warning around direct `Models.generate_content()` calls even when the project is not using tools. The SDK repository currently documents AFC behavior changes and an issue tracks the warning when tools are absent.

This project does not use function calling or tools.

Therefore the notebook explicitly sets:

```python
automatic_function_calling=types.AutomaticFunctionCallingConfig(
    disable=True
)
```

This keeps the Gemini call as a normal text-generation request and prevents the irrelevant AFC warning.

---

# 15. Step 11 — Reviewer edge-case test

After the three live drafts are generated, the notebook asks:

```text
Enter the review number to revise (1, 2, or 3):
```

Only the discrete values **1, 2, or 3** are accepted.

If the reviewer enters something else, the notebook asks again.

This is an input-validation edge case.

The reviewer then enters an optional additional comment.

There is no artificial small character limit in the notebook.

---

# 16. Step 12 — Gemini uses both inputs

The final generation step gives Gemini:

1. the selected original review;
2. the reviewer/customer's additional comment.

Gemini then produces another response draft.

This demonstrates that the project can handle a new piece of information without changing the original Critical-selection logic.

The final output is still a draft.

---

# 17. How to evaluate the notebook

An evaluator can inspect the notebook from top to bottom.

### Data stage

Check:

- shape;
- columns;
- `info()`;
- `describe()`;
- missing values;
- duplicates;
- rating validity.

Expected key values:

```text
Rows: 3,600
Columns: 11
Exact duplicates: 0
Critical reviews: 687
Critical rate: 19.08%
```

### EDA stage

Check whether the notebook includes:

- rating distribution;
- written inference;
- clear explanation of why rating is the triage target.

### Rule-based stage

Check:

```text
Critical = rating 1 or 2
```

and the complaint dictionary.

### Gemini stage

Enter a valid Gemini API key at runtime.

The notebook should then:

1. create the Gemini client;
2. generate three live drafts;
3. display the three drafts;
4. ask for 1, 2, or 3;
5. reject invalid review numbers;
6. accept an additional comment;
7. generate a revised draft.

### Security stage

The evaluator should confirm that:

- no API key is hard-coded;
- the key is not printed;
- the key is not stored in the notebook;
- the project only generates drafts.

---

# 18. Reproducibility and responsibility

The analytical portion is deterministic:

- the Critical rule is fixed;
- the complaint dictionary is fixed;
- the top-three selection order is fixed.

Gemini generation itself is probabilistic, so exact wording can vary between runs.

The project therefore separates reproducible analytical decisions from probabilistic language generation.

---

# 19. Limitations

The main limitations are:

1. The complaint dictionary can miss unseen wording.
2. Keyword matching does not understand full context or negation.
3. The dataset is a fixed historical review dataset.
4. Gemini responses can vary between runs.
5. Human review is required before sending any real response.
6. The notebook does not send email automatically.

---

# 20. Future scope

Only after the current assignment is approved, a production version could add:

- a web UI;
- secure server-side API-key handling;
- human approval workflow;
- CRM integration;
- multilingual responses;
- richer complaint taxonomy;
- response-quality monitoring;
- labelled-data-based ML classification.

These are future enhancements, not part of the current rule-based assessment.

---

## 21. Environment

The notebook uses:

- Python 3
- Pandas
- Matplotlib
- `google-genai`
- `requests` for the optional public-dataset download

The Gemini API key is supplied at runtime with hidden input.

---

## Final project statement

The project can be summarized in one sentence:

> **Python transparently identifies and prioritizes Critical customer feedback using explicit rules, while Gemini converts the selected customer complaints into concise, empathetic response drafts for human review.**


This is live in Github link:
