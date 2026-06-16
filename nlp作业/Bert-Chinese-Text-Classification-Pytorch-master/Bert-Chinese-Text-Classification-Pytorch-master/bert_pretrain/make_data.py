import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 项目数据目录
data_dir = Path("THUCNews/data")

# 原始数据文件名
csv_path = data_dir / "ChnSentiCorp_htl_all.csv"

# 读取数据
df = pd.read_csv(csv_path)

# 看一下列名，正常应该有 label 和 review
print("原始列名：", df.columns.tolist())

# 只保留 label 和 review 两列
df = df[["label", "review"]]

# 删除空值
df = df.dropna()

# label 转成整数，review 转成字符串
df["label"] = df["label"].astype(int)
df["review"] = df["review"].astype(str)

# 去掉文本里的换行、tab，避免破坏 train.txt 格式
df["review"] = df["review"].str.replace("\n", " ", regex=False)
df["review"] = df["review"].str.replace("\r", " ", regex=False)
df["review"] = df["review"].str.replace("\t", " ", regex=False)

# 去掉空评论
df = df[df["review"].str.strip() != ""]

# 打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 先划分训练集 80%，临时集 20%
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 再把临时集分成验证集 10%，测试集 10%
dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

# 保存函数：格式是 评论文本 + tab + 标签数字
def save_txt(dataframe, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            text = str(row["review"]).strip()
            label = int(row["label"])
            f.write(f"{text}\t{label}\n")

save_txt(train_df, data_dir / "train.txt")
save_txt(dev_df, data_dir / "dev.txt")
save_txt(test_df, data_dir / "test.txt")

# 写类别文件，注意顺序：第0类 negative，第1类 positive
with open(data_dir / "class.txt", "w", encoding="utf-8") as f:
    f.write("negative\n")
    f.write("positive\n")

print("数据处理完成！")
print("训练集数量：", len(train_df))
print("验证集数量：", len(dev_df))
print("测试集数量：", len(test_df))
print("标签分布：")
print(df["label"].value_counts())