# 语义表征引导的补丁验证实验

这是基于本仓库阶段计划新写的参考实现。原实验代码、模型修订号、395 个缺陷的逐候选数据及历史日志未随稿件提交。这里的默认超参数、提示词、分类器和成本模型属于本实现的选择。历史统计不作为程序输入或期望输出。新实验需要保存真实候选、标签、配置和运行结果后才能引用。

## 从缺陷到候选

在隔离的实验机安装 Defects4J、项目所需 JDK、Python 3.10+ 和 requirements.txt。每个缺陷提供一行 JSON，字段为 project、bug、split、file、start_line、end_line、original、n_tests，可提供 prompt、encoding。位置为 buggy checkout 中闭区间的行号，original 为完整原片段，patch 表示该处的替换文本。本生成/验证入口支持单个连续修改区域；历史多文件候选可在原验证程序测量后走导入路径。固定 Defects4J 版本与硬件，保存命令输出。文件标识和人工标注不从测试项目训练得到。

```bash
cd experiments
python -m pip install -r requirements.txt
python prepare.py generate bugs.jsonl candidates.jsonl --revision "$GENERATOR_SHA" --samples 40
python prepare.py validate candidates.jsonl validation.jsonl
python prepare.py encode adjudicated.jsonl cache.npz --encoder-revision "$ENCODER_SHA" --decoder-revisions "$SMALL_SHA" "$LARGE_SHA"
```

生成阶段按缺陷精确去重，记录生成分数；验证阶段每个候选使用独立的 buggy checkout，记录编译与全套测试的耗时和输出。通过测试的候选 correct 留空，须按原协议独立核对功能正确性、解决分歧并保存标注依据。请在无凭据的隔离实验环境运行测试。

## 原始候选导入

已有候选可直接整理为 adjudicated.jsonl，每项须含 id（全局唯一字符串）、project、bug、split、original、patch、gen_score（越大越优）、n_tests、seconds、passed、correct。每个项目只允许分配到 train、cal、test 之一。cal 来自开发项目，测试项目独立。对生成零候选的缺陷，请单独核对缺陷清单并计入全缺陷指标，当前脚本只对清单中至少有一条记录的缺陷求值。测试集和候选池须与论文逐项对齐后才可引用结果。

缓存 NPZ 需要 ids、diff（N×D）、prefix（N×10×H）、seconds（每个候选的表征提取秒数）。默认 diff 为 UniXcoder 原、改、改减原向量拼接；prefix 为两个冻结 Qwen 模型在 10% 至 100% 相对位置的末层隐藏状态。在因果模型的一次前向计算中读取这些位置。保存真实修订哈希。短补丁可能重复取到同一位置。

## 训练、消融及验证调度

```bash
OPENBLAS_NUM_THREADS=1 python evaluate.py adjudicated.jsonl cache.npz results --budgets 30 60 120 --reward 0.25 --threshold 0.5 --k 10
python evaluate.py adjudicated.jsonl cache.npz results_plausible --budgets 30 60 120 --reward 0.25 --threshold 0.5 --k 10 --train-on plausible
python check_pipeline.py
```

命令参数为示例，在开发集确定预算、奖励权重、K 和阈值后冻结。默认五个按缺陷自助抽样的轻量模型。每个种子结果及 ensemble 均输出。仅差异、仅多位置、末位置融合、平均汇聚、注意力融合、移除排序损失共用训练划分。仅多位置为受 AutoProbe 启发的固定层基线；原方法包含跨层选择。真实数据上应额外记录 GPU 型号、模型初始化和各阶段墙钟时间。运行另一项目划分时用新 manifest 和独立输出目录，禁止复用测试项目拟合的模型。

rank_by_bug.csv、rank_all_bugs.csv 和 quality.csv 给出排名、Top-k 及通过测试子集的 AUC。reliability.csv 和 quality.csv 给出校准分箱和 Brier。predictions.csv 可核查耗时预测、模型分歧和逐候选误差。schedule_by_bug.csv 保存执行顺序、终止时未完成的候选、最终返回及真实标签评估结果；schedule.csv 汇总全缺陷成功率、返回覆盖率、条件正确率、保留率、错误返回和总时间。paired.csv 报告以缺陷为单位的配对 bootstrap 差值、区间、改善和退化数量。逐项目分析可按两个 by_bug 表的 project 分组。settings.json 记录运行参数和耗时预测 MAE。

成本模型仅用训练项目已知的补丁规模和测试数量拟合。每次选择仅用预估成本和此前已完成的开销，实际耗时用于回放执行和扣除预算。预算不足以完成一次验证时无法取得测试结果。联合策略与概率/成本策略共用其余决策规则。不确定性奖励来自校准后的模型分歧。完成一次测试后更新同一缺陷的成本倍率，并未用事后正确标签更新未执行候选。回放时间由提取时间和实际编译测试时间构成，未计模型加载、训练、生成、轻量评分或调度端到端耗时。对所有使用语义打分的策略统一收取完整特征开销；best 返回规则也需要评分。对速度作最终结论需另做实机计时。

check_pipeline.py 仅用合成数组验证指标分母、预算边界、错误返回与代码路径，不产生论文证据。
