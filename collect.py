import os
import json
import pandas as pd

# 参数选择范围
years = [2020, 2021, 2022]
ids = [50, 300, 500]
models = ["Attention", "CNN",  "GRU", "preTCN", "FuturesNet"]

results_list = []

# 迭代所有的 year, id 和 model 组合
for year in years:
    for data_id in ids:
        for model in models:
            # 组合成文件路径
            dir_path = f"save/{year}_{data_id}/{model}"
            file_path = os.path.join(dir_path, "results.json")

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取 json 文件
                with open(file_path, "r") as f:
                    data = json.load(f)

                # 提取指标，并添加模型信息
                results = {
                    "year": year,
                    "id": data_id,
                    "model": model,
                    "best_acc": data.get("best_acc"),
                    "mae": data.get("mae"),
                    "mse": data.get("mse"),
                    "rmse": data.get("rmse"),
                    "r2": data.get("r2"),
                    "val_loss": data.get("val_loss"),
                    "sharp_value": data.get("sharp_value")
                }

                results_list.append(results)

df = pd.DataFrame(results_list)
output_csv = "model_summary.csv"
df.to_csv(output_csv, index=False)

print(f"Results have been written to {output_csv}")