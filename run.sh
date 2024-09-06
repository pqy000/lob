#!/bin/bash

# 检查是否有 log 目录，如果没有则创建
if [ ! -d "log" ]; then
  mkdir log
  echo "Created log directory."
fi

# 从命令行参数获取 year, id 和 model 列表
years=($1)
ids=($2)
models=($3)

# 循环遍历所有 id, year 和 model 组合
for year in "${years[@]}"
do
  for id in "${ids[@]}"
  do
    for model in "${models[@]}"
    do
      # 输出日志文件路径
      log_file="log/output_${id}_${year}_${model}.log"

      echo "Running id=$id, year=$year, model=$model"
      nohup python3 main.py --id $id --year $year --model $model > $log_file 2>&1 &
      echo "Started process for id=$id, year=$year, model=$model in background with PID $!"
    done
  done
done

wait
echo "All processes completed."
