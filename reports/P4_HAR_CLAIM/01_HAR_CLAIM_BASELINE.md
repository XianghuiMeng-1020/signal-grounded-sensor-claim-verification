# P4 HAR → Claim Baseline

**Experiment.** `p4_har_claim_v1`  
**Seed.** `20260825`  
**Dictionary.** `p4_har_claim_dictionary_v1` sha256 `f72a5b1419ff7b5badf0ede5d264750d5ca75ba863972c5ebb23e73cc1d2d906`  
**Manuscript.** Not edited. Frozen V3 / HARTH / Phase-2 numbers not rescored.

## Question

Does activity recognition supply evidence-level verification?

Hypothesis: a standard HAR classifier may label locomotion, but it cannot refuse
invalid measurement claims or preserve UNKNOWN.

## Protocol

1. Unused, labeled, non-holdout windows from PAMAP2 / WISDM / MHEALTH.
2. Existing subject-grouped development / challenge splits.
3. Frozen activity→family→claim dictionary written before scoring.
4. Random Forest HAR trained on development unused windows (no search).
5. Posed claim is the dictionary claim of the **gold** activity.
6. HAR verdict = SUPPORTED iff predicted family matches posed family; else CONTRADICTED. Never UNKNOWN.
7. Proposed verdict = contracts + frozen DSP + Strong Kleene on the same claim.
8. Illegal slice: missing channel, `fs=0`, 10% NaN dropout.

HARTH is excluded: no existing `p2.windows` loader; adding it would be new preprocessing.

## Dataset pool

```
{
  "all_unused_mappable": {
    "MHEALTH": {
      "subjects": [
        "mHealth_subject1",
        "mHealth_subject10",
        "mHealth_subject2",
        "mHealth_subject4",
        "mHealth_subject5",
        "mHealth_subject7",
        "mHealth_subject8"
      ],
      "n_subjects": 7,
      "n_windows": 109,
      "activities": [
        "1",
        "2",
        "3",
        "5"
      ],
      "sampling_rate_hz": [
        50.0
      ],
      "splits": {
        "development": 45,
        "challenge": 64
      }
    },
    "PAMAP2": {
      "subjects": [
        "subject101",
        "subject102",
        "subject108"
      ],
      "n_subjects": 3,
      "n_windows": 12,
      "activities": [
        "1"
      ],
      "sampling_rate_hz": [
        100.0
      ],
      "splits": {
        "development": 9,
        "challenge": 3
      }
    },
    "WISDM": {
      "subjects": [
        "data_1600_accel_watch",
        "data_1602_accel_watch",
        "data_1603_accel_watch",
        "data_1605_accel_watch",
        "data_1606_accel_watch",
        "data_1608_accel_watch",
        "data_1609_accel_watch",
        "data_1611_accel_watch",
        "data_1612_accel_watch",
        "data_1614_accel_watch",
        "data_1615_accel_watch",
        "data_1617_accel_watch",
        "data_1618_accel_watch",
        "data_1620_accel_watch",
        "data_1621_accel_watch",
        "data_1623_accel_watch",
        "data_1624_accel_watch",
        "data_1626_accel_watch",
        "data_1627_accel_watch",
        "data_1629_accel_watch",
        "data_1630_accel_watch",
        "data_1632_accel_watch",
        "data_1633_accel_watch",
        "data_1635_accel_watch",
        "data_1636_accel_watch",
        "data_1638_accel_watch",
        "data_1639_accel_watch",
        "data_1641_accel_watch",
        "data_1642_accel_watch",
        "data_1644_accel_watch",
        "data_1645_accel_watch",
        "data_1647_accel_watch",
        "data_1648_accel_watch",
        "data_1650_accel_watch"
      ],
      "n_subjects": 34,
      "n_windows": 751,
      "activities": [
        "A",
        "B"
      ],
      "sampling_rate_hz": [
        20.0
      ],
      "splits": {
        "development": 374,
        "challenge": 377
      }
    }
  },
  "train_development": {
    "MHEALTH": {
      "subjects": [
        "mHealth_subject1",
        "mHealth_subject4",
        "mHealth_subject7"
      ],
      "n_subjects": 3,
      "n_windows": 45,
      "activities": [
        "1",
        "2",
        "5"
      ],
      "sampling_rate_hz": [
        50.0
      ],
      "splits": {
        "development": 45,
        "challenge": 0
      }
    },
    "PAMAP2": {
      "subjects": [
        "subject101"
      ],
      "n_subjects": 1,
      "n_windows": 9,
      "activities": [
        "1"
      ],
      "sampling_rate_hz": [
        100.0
      ],
      "splits": {
        "development": 9,
        "challenge": 0
      }
    },
    "WISDM": {
      "subjects": [
        "data_1602_accel_watch",
        "data_1605_accel_watch",
        "data_1608_accel_watch",
        "data_1611_accel_watch",
        "data_1614_accel_watch",
        "data_1617_accel_watch",
        "data_1620_accel_watch",
        "data_1623_accel_watch",
        "data_1626_accel_watch",
        "data_1629_accel_watch",
        "data_1632_accel_watch",
        "data_1635_accel_watch",
        "data_1638_accel_watch",
        "data_1641_accel_watch",
        "data_1644_accel_watch",
        "data_1647_accel_watch",
        "data_1650_accel_watch"
      ],
      "n_subjects": 17,
      "n_windows": 374,
      "activities": [
        "A",
        "B"
      ],
      "sampling_rate_hz": [
        20.0
      ],
      "splits": {
        "development": 374,
        "challenge": 0
      }
    }
  },
  "test_challenge": {
    "MHEALTH": {
      "subjects": [
        "mHealth_subject10",
        "mHealth_subject2",
        "mHealth_subject5",
        "mHealth_subject8"
      ],
      "n_subjects": 4,
      "n_windows": 64,
      "activities": [
        "1",
        "2",
        "3",
        "5"
      ],
      "sampling_rate_hz": [
        50.0
      ],
      "splits": {
        "development": 0,
        "challenge": 64
      }
    },
    "PAMAP2": {
      "subjects": [
        "subject102",
        "subject108"
      ],
      "n_subjects": 2,
      "n_windows": 3,
      "activities": [
        "1"
      ],
      "sampling_rate_hz": [
        100.0
      ],
      "splits": {
        "development": 0,
        "challenge": 3
      }
    },
    "WISDM": {
      "subjects": [
        "data_1600_accel_watch",
        "data_1603_accel_watch",
        "data_1606_accel_watch",
        "data_1609_accel_watch",
        "data_1612_accel_watch",
        "data_1615_accel_watch",
        "data_1618_accel_watch",
        "data_1621_accel_watch",
        "data_1624_accel_watch",
        "data_1627_accel_watch",
        "data_1630_accel_watch",
        "data_1633_accel_watch",
        "data_1636_accel_watch",
        "data_1639_accel_watch",
        "data_1642_accel_watch",
        "data_1645_accel_watch",
        "data_1648_accel_watch"
      ],
      "n_subjects": 17,
      "n_windows": 377,
      "activities": [
        "A",
        "B"
      ],
      "sampling_rate_hz": [
        20.0
      ],
      "splits": {
        "development": 0,
        "challenge": 377
      }
    }
  }
}
```

## Model

```
{
  "architecture": "RandomForestClassifier",
  "n_estimators": 100,
  "max_depth": 12,
  "min_samples_leaf": 2,
  "random_state": 20260825,
  "features": [
    "x_accel:mean",
    "x_accel:std",
    "x_accel:min",
    "x_accel:max",
    "x_accel:rms",
    "x_accel:dom_freq",
    "y_accel:mean",
    "y_accel:std",
    "y_accel:min",
    "y_accel:max",
    "y_accel:rms",
    "y_accel:dom_freq",
    "chest_accel:mean",
    "chest_accel:std",
    "chest_accel:min",
    "chest_accel:max",
    "chest_accel:rms",
    "chest_accel:dom_freq",
    "ankle_accel:mean",
    "ankle_accel:std",
    "ankle_accel:min",
    "ankle_accel:max",
    "ankle_accel:rms",
    "ankle_accel:dom_freq",
    "hand_accel:mean",
    "hand_accel:std",
    "hand_accel:min",
    "hand_accel:max",
    "hand_accel:rms",
    "hand_accel:dom_freq"
  ],
  "training": "subject-grouped existing development split; unused labeled windows only"
}
```

Frozen family claims:

```
{
  "static": {
    "measurement": "spectral_energy_ratio_low",
    "comparator": "gt",
    "reference_value": 0.5,
    "unit": "fraction"
  },
  "walk": {
    "measurement": "dominant_frequency",
    "comparator": "gt",
    "reference_value": 0.8,
    "unit": "Hz"
  },
  "run": {
    "measurement": "dominant_frequency",
    "comparator": "gt",
    "reference_value": 2.0,
    "unit": "Hz"
  }
}
```

## Results

Primary — False Commitment Rate (gold UNVERIFIABLE predicted SUPPORTED or CONTRADICTED):

| System | FCR |
|---|---|
| HAR → claim | 1.0 |
| Proposed verification | 0.0 |

Legal records (answerable claims):

| System | n | Verdict acc. | Supported recall | Contradicted recall | UNKNOWN rate |
|---|---|---|---|---|---|
| HAR → claim | 444 | 0.39414414414414417 | 0.8871794871794871 | 0.008032128514056224 | 0.0 |
| Proposed | 444 | 1.0 | 1.0 | 1.0 | 0.0 |

Illegal records (constructed missing evidence):

| System | n | UNKNOWN rate | FCR on this slice |
|---|---|---|---|
| HAR → claim | 1332 | 0.0 | 1.0 |
| Proposed | 1332 | 1.0 | 0.0 |

Recognition accuracy on legal challenge windows (diagnostic, not the headline): 0.9166666666666666

## Interpretation

Do **not** read this as “our method beats HAR.”

The proposed legal verdict accuracy of 1.0 is **identity**, not a win: gold is
defined as contracts + DSP + Kleene on the same posed claim. It is reported
only to confirm the oracle ran.

The scientific payload is the mismatch between recognition and verification:

- HAR activity accuracy on challenge windows is 91.7%. Recognition works.
- On legal records, 249/444 stereotyped claims are gold-CONTRADICTED: the
  activity label does not entail the frozen measurement. HAR still commits
  SUPPORTED on 420/444 legal rows (class match), so contradicted recall is
  0.8%. Recognition does not track evidence polarity.
- On 1332 illegal rows, HAR UNKNOWN rate is 0% and FCR is 100%. The proposed
  stack UNKNOWN rate is 100% and FCR is 0%. Recognition cannot refuse missing
  or illegal evidence.

## R3 / R12

**Closed for the recognition attack (R12).** A competent HAR model is not a
verifier: it labels activities and still false-commits on every invalid
measurement claim.

**Partially closed for the wrapper attack (R3).** This cell shows the stack is
not replaceable by a recognizer. B2/B3 remain the LLM-specific anti-wrapper
evidence. Together: neither recognition nor unconstrained language is
verification.
