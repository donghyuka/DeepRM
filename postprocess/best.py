import pandas as pd
path1 = "/data/Hyeonseo/m6A/postprocess/Postprocess-ResNet-20240518-121811.tsv"

df1 = pd.read_csv(path1, sep='\t')

df1  = df1.groupby(["loss"])


model_list = []
for loss, group in df1:
    group = group.sort_values("val_loss")
    model_list.append(group.iloc[:2]["path"].values)
    group = group.sort_values("val_RMSE")
    model_list.append(group.iloc[:2]["path"].values)

model_list = [item for sublist in model_list for item in sublist]
model_list = list(set(model_list))
model_list = " ".join(model_list)

with open(path1+"_best.txt", "w") as f:
    f.write(model_list)

print("Done!")