# Failure modes on test150 (summary)

This is a summary of the failure analysis, not a reproduction of the original
paper tables. Raw images and full model outputs are not redistributed.

All 150 samples stay in the denominator; truncated, invalid, or missing outputs
score zero. The examples below are post-hoc explanations and were not used to
re-tune the frozen setting.

## Aggregate

- Q1 original: 15 truncated, 0 invalid; Q1 frozen: 13 truncated, 0 invalid.
- 12 samples remain truncated in both settings.
- The frozen setting recovers 3 cases and introduces 1 new truncation.
- Paddle: 0 truncated, 2 invalid, 148/150 valid.

## Typical failure modes

1. **Repetition loop → truncation**
   - The model starts repeating empty cells / `<0.001` / blank rows and reaches
     the 4096-token limit.
   - In the new setting, one previously complete sample (PMC5577821_005_00)
     turns into a 4096-token truncation; three other cases recover.
   - This is not solved by the visual-budget change alone.

2. **Structure mismatch with merged cells**
   - Rowspan/colspan expansion is wrong even when text is readable; e.g. a
     semester cell spans too many rows, or a two-level header becomes an extra
     empty row.
   - This is the dominant error type in the Chinese dev probe as well.

3. **Cell content errors after correct structure**
   - Digits, signs, units, citation markers, or superscripts differ.
   - A structure-perfect HTML can still have wrong numbers; TEDS is not business
     accuracy.

4. **Long outputs**
   - A 32×13 table can exceed 4096 tokens even if the structure is correct; this
     is an output-budget limit, not a language issue.

## Examples (scores only)

| Sample | Q1 original | Q1 frozen | Paddle | Main issue |
|---|---:|---:|---:|---|
| PMC5577821_005_00 | 0.770 | 0.000 | 0.670 | new truncation / repetition |
| PMC3660179_007_00 | 0.000 | 0.816 | 0.810 | recovers; content mismatch |
| PMC6068535_008_00 | 0.000 | 0.917 | 0.655 | recovers; structure near-perfect |
| PMC4304202_001_00 | 0.000 | 0.977 | — | recovers; content mismatch |

## What this means

- The visual-budget improvement is real but partial: it fixes some failures and
  adds one new one.
- Structure and repetition remain the main blockers, not the base model's
  ability to read English text.
- The sealed test is not used to tune these cases; new work needs a separate
  development set.
