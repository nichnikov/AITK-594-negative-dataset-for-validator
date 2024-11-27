import os
import pandas as pd

df = pd.read_feather(os.path.join("datasets", "bss_2021.feather"))

print(df)
df_dicts = df.to_dict(orient="records")
for d in df_dicts[:15]:
    print("InQuery:", d["InQuery"], "\nClearQuery:", d["ClearQuery"], "\nClearAnswer:", d["ClearAnswer"], "\nlabel:", d["label"], "\n\n")


# print(df_dicts[:5])
# df.to_csv(os.path.join("datasets", "bss_2021.csv"), sep="\t", index=False)