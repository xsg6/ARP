# 语义表征引导的补丁验证实验

这是基于本仓库阶段计划新写的参考实现。原实验代码、模型修订号、395 个缺陷的逐候选数据及历史日志未随稿件提交。这里的默认超参数、提示词、分类器和成本模型属于本实现的选择。历史统计不作为程序输入或期望输出。新实验需要保存真实候选、标签、配置和运行结果后才能引用。

## 从缺陷到候选

在隔离的实验机安装 Defects4J、项目所需 JDK、Python 3.10+ 和 requirements.txt。每个缺陷提供一行 JSON，字段为 project、bug、split、file、start_line、end_line、original、n_tests，可提供 prompt、encoding。位置为 buggy checkout 中闭区间的行号，original 为完整原片段，patch 表示该处的替换文本。本生成/验证入口支持单个连续修改区域；历史多文件候选可在原验证程序测量后走导入路径。固定 Defects4J 版本与硬件，保存命令输出。文件标识和人工标注不从测试项目训练得到。

```bash
cd experiments
python -m pip install -r requirements.txt
python prepare.py generate bugs.jsonl candidates.jsonl --revision "$GENERATOR_SHA" --samples 50
python prepare.py validate candidates.jsonl validation.jsonl
python prepare.py encode adjudicated.jsonl cache.npz --encoder-revision "$ENCODER_SHA" --decoder-revisions "$SMALL_SHA" "$LARGE_SHA"
```

生成阶段按缺陷精确去重，记录生成分数；验证阶段每个候选使用独立的 buggy checkout，记录编译与全套测试的耗时和输出。通过测试的候选 correct 留空，须按原协议独立核对功能正确性、解决分歧并保存标注依据。请在无凭据的隔离实验环境运行测试。

## 原始候选导入

已有候选可直接整理为 adjudicated.jsonl，每项须含 id（全局唯一字符串）、project、bug、split、original、patch、gen_score（越大越优）、n_tests、seconds、passed、correct。同一缺陷的候选须在同一划分，测试项目与开发项目隔离。cal 来自开发项目，测试项目独立。正式评估请传入 --roster bugs.jsonl，对零候选缺陷按 MRR、Top-k、最终成功均为零计入分母；缺省仅评估清单中有候选的缺陷。测试集和候选池须与论文逐项对齐后才可引用结果。

缓存 NPZ 需要 ids、diff（N×D）、prefix（N×10×H）、seconds（每个候选的表征提取秒数）。默认 diff 为 UniXcoder 原、改、绝对差向量拼接；prefix 为两个冻结 Qwen 模型在 10% 至 100% 相对位置的末层隐藏状态。在因果模型的一次前向计算中读取这些位置。保存真实修订哈希。短补丁可能重复取到同一位置。

## 训练、消融及验证调度

```bash
OPENBLAS_NUM_THREADS=1 python evaluate.py adjudicated.jsonl cache.npz results --budgets 30 60 120 --reward 0.25 --threshold 0.5 --k 10
python evaluate.py adjudicated.jsonl cache.npz results_all --budgets 30 60 120 --reward 0.25 --threshold 0.5 --k 10 --train-on all
python report.py results
python check_pipeline.py
```

命令参数为示例，在开发集确定预算、奖励权重、K 和阈值后冻结。正式后端默认训练五个独立初始化的轻量模型。每个种子结果及 ensemble 均输出。仅差异、仅多位置、末位置融合、平均汇聚、注意力融合、移除排序损失共用训练划分。仅多位置为受 AutoProbe 启发的固定层基线；原方法包含跨层选择。真实数据上应额外记录 GPU 型号、模型初始化和各阶段墙钟时间。运行另一项目划分时用新 manifest 和独立输出目录，禁止复用测试项目拟合的模型。

rank_by_bug.csv、rank_all_bugs.csv 和 quality.csv 给出排名、Top-k 及通过测试子集的 AUC。reliability.csv 和 quality.csv 给出校准分箱和 Brier。predictions.csv 可核查耗时预测、模型分歧和逐候选误差。schedule_by_bug.csv 保存执行顺序、终止时未完成的候选、最终返回及真实标签评估结果；schedule.csv 汇总全缺陷成功率、返回覆盖率、条件正确率、保留率、错误返回和总时间。paired.csv 报告以缺陷为单位的配对 bootstrap 差值、区间、改善和退化数量。逐项目分析可按两个 by_bug 表的 project 分组。settings.json 记录运行参数、输入校验和与耗时预测 MAE。report.py 生成逐项目汇总、发生修复得失的具体缺陷清单、预算曲线和可靠性曲线。

成本模型仅用训练项目已知的补丁规模和测试数量拟合。每次选择仅用预估成本和此前已完成的开销，实际耗时用于回放执行和扣除预算。预算不足以完成一次验证时无法取得测试结果。联合策略与概率/成本策略共用其余决策规则。不确定性奖励来自校准后的模型分歧。完成一次测试后更新同一缺陷的成本倍率，并未用事后正确标签更新未执行候选。回放时间由提取时间和实际编译测试时间构成，未计模型加载、训练、生成、轻量评分或调度端到端耗时。对所有使用语义打分的策略统一收取完整特征开销；best 返回规则也需要评分。对速度作最终结论需另做实机计时。

check_pipeline.py 仅用合成数组验证指标分母、预算边界、错误返回与代码路径，不产生论文证据。

## 对应原小论文各研究问题

正式评估用默认的 paper 后端。scorer.py 实现每个解码器独立的 tanh 注意力、512 维 GELU 评分头、0.2 dropout、加权分类损失、0.2 间隔与 0.5 排序损失权重、同缺陷随机及困难负例、AdamW 和验证 AUC 早停。注意力中间宽度默认 64，为原文未指定的实现选择。linear 后端仅用于快速流程检查。旧表按可修复缺陷计算 MRR、Top-k 和首个正确候选的中位排名；使用 rank_repairable.csv。新增全缺陷指标在 rank_all_bugs.csv。两者分母不同。

1. RQ1 与 RQ4。正式提取时在两个主解码器之后增加 Qwen/Qwen2.5-Coder-0.5B 并提供三个对应的 --decoder-revisions。三个模型顺序为 1.5B、7B、0.5B。仅 0.5B 探针使用第三组状态。输入包含 buggy_context、局部 original、failure 和 patched_context 时可完整保留方法级上下文。默认用通过测试的候选训练；全候选训练请显式传 --train-on all。原文用 Closure、Mockito、Time 开发，在这三个项目内按缺陷 80/20 划分训练/验证，Lang、Math、Chart 留作测试。重建划分须说明原始种子未知。
2. RQ2。random_repairable.csv 使用每个缺陷 1000 次随机排列。原文还与 ODS 比较。按 ODS 作者仓库 ASSERT-KTH/ODSExperiment 的说明，用 Java 8 构建 SpoonLabs/coming。用 prepare.py validate 的 --sources 保存完整修改前后源文件。执行 python ods.py adjudicated.jsonl coming-jar-with-dependencies.jar scored.jsonl，再将 scored.jsonl 作为 evaluate.py 的 manifest。ODS 仅在训练项目的已标注可行候选上拟合，用正确类概率排序；保存 Coming 版本及输入。已有逐候选 ODS 分数也可按 ID 写入 ods_score。
3. RQ3。fixed_k_cost_by_bug.csv 记录穷举、生成概率 Top-20、固定 K 与阈值及各消融的验证次数和耗时。与后续的“预算内择优返回”分开。静态验证使用开发集确定的 K、阈值，报告选择策略的正确候选保留率。
4. RQ5。injection.py 从训练集已标注正确的补丁形成全局语义原型，训练投影与门控后在冻结解码器输出层之前注入。原文未公布注入训练细节，本实现选择正确目标的 teacher-forced token 损失、128 维原型空间。先运行 python injection.py train adjudicated.jsonl cache.npz adapter.pt --revision "$GENERATOR_SHA"，再用相同解码参数和 50 次采样分别运行 prepare.py generate，有注入组增加 --adapter adapter.pt。两组均独立验证和标注，保留每个去重补丁的 draw_count，再运行 python injection.py compare baseline_labeled.jsonl injected_labeled.jsonl bugs.jsonl rq5.json --draws 50。只有确认每个缺陷都完成规定次数的采样时才能使用该分母。RQ5 候选池与 RQ1–RQ4 不混用。

正式运行可加 --select-fixed，在开发缺陷中搜索达到至少 95% 条件保留率且实测验证费用最低的 K 与阈值，二者记入 settings.json。fixed_k_cost.csv 汇总各分支耗时与相对穷举节省。auc_ci.csv 为逐缺陷重采样区间。缓存旁的 .json 记录各骨干前向耗时、特征字节数及 GPU 峰值分配量。论文未提供原始训练集、外部标注及精确运行设置，这些代码不能逐位复现历史表格。check_models.py 使用随机初始化的小模型检查实际库调用、特征维度与可训练注入路径。
