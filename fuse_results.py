import glob, pandas as pd
df = pd.concat(pd.read_csv(f) for f in glob.glob("run_*csv"))
df.to_csv("all_results.csv", index=False)