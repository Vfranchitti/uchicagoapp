# Base rates and false alarms

- Expected leaks among the 40 undug sites at 1 in 8: **5**.
- Five blind digs find **0.62** leaks on average; there is a **51%** chance they find none. Cost per leak found: **$64,000**.
- A rule that catches 80% of leaks and clears 90% of sound sites is right only **53%** of the time it says "leak" at 1 in 8, but would look **89%** right on a sample that is 50% leaks, like the dug sites. Applied to all 40 sites it would flag about 4 leaks and 3.5 false alarms.
- Even a perfect score on the dug sites says little: 3 of 3 leaks caught is consistent (95% CI) with true sensitivity as low as **29%**; 6 of 6 correct is consistent with accuracy as low as **54%**. A coin flip gets all 6 right 1.6% of the time.

Full grid: `base_rate_ppv_scenarios.csv`.
