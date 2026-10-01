# Support Triage AI：中文运行与学习指南

## 项目定位

这是一个供小型软件团队使用的客户咨询分类助手。产品收到的消息可能是账单疑问、账号登录问题、故障报告或功能建议；程序学习这些文本的区别，推荐处理队列和初次回复模板。

这是独立的产品场景，可用于软件产品、在线工具或订阅服务。它展示的文本分类、输入校验、可解释预测和人工复核能力，也可以迁移到其他行业。

技术难度保持入门级：Python、TF-IDF、逻辑回归和 Streamlit。无需显卡、模型 API 或付费服务。模型只面向英文咨询。

## 安装与运行

在项目目录打开终端，使用 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Windows 使用 `py -3.12 -m venv .venv` 创建环境，使用 `.venv\Scripts\Activate.ps1` 激活。打开终端提示的网址，通常是 `http://localhost:8501`。首次安装依赖需要网络，之后的模型训练和预测在本地执行。

## 四类咨询

| 类别 | 示例 | 初次回复收集的信息 |
|---|---|---|
| Billing & subscriptions | I was charged twice for my monthly subscription. | 发票编号、希望核实的账单问题 |
| Account access | I forgot my password and cannot log in. | 登录失败的步骤和错误文字，不收集密码或验证码 |
| Bug report | The app crashes every time I upload a file. | 复现步骤、预期行为、实际行为、版本 |
| Feature request | Could you add a dark mode to the application? | 用户想完成的任务、建议功能的用途 |

不确定的消息进入 Manual review。程序不执行退款、账号解锁或邮件发送。回复是可编辑模板，不是大模型生成的答案。

## 三分钟演示

1. 选择 Billing，点击 Analyze inquiry，展示类别和草稿。
2. 查看贡献较大的词，解释模型如何利用文本特征。
3. 选择 Unclear inquiry，展示人工复核。
4. 在 Batch workspace 点击 Analyze sample inbox，查看五条结果并导出。
5. 打开评估报告，说明演示数据是合成数据，尚未用于真实客服。

样例收件箱使用训练集中已有的例句，便于稳定演示。它不是评估集，不能用它证明泛化能力。

## 理解核心代码

建议依次阅读：

1. `data/train.csv`：每条消息和对应标签。
2. `support_ai/model.py` 的 `train_model`：训练管道。
3. 同文件的 `classify`：预测、人工复核和词语贡献。
4. `support_ai/batch.py`：CSV 校验和逐行错误处理。
5. `app.py`：界面如何调用这些功能。
6. `support_ai/evaluate.py` 和 `tests/test_project.py`：评估和测试。

TF-IDF 将词和二元词组转换为数字特征，逻辑回归从标注数据学习每类的权重。这里“逻辑回归”用于分类。单条预测显示的模型分数没有经过校准，不能理解为经过验证的正确率。

当最高分不足、前两类分数接近、已知词太少或可能不是英文时，程序建议人工复核。这些规则不是完整的异常检测系统，仍然可能漏掉复杂问题。否定句、多意图和陌生表达尤其容易出错。

## 测试与评估

```bash
python -m pytest -q
python -m support_ai.evaluate
```

全部数据为人为编写的合成消息：80 条训练、16 条验证、24 条测试、10 条复核挑战。训练只读取 train.csv，其他数据不参与词表或分类器学习。阈值在评估前固定，没有用测试集选最优值。

`reports/evaluation.md` 指标说明：

- Raw accuracy：最高分标签的整体准确率，含之后被转人工的样本。
- Macro F1：先对各类别计算 F1，再取平均。
- Routing coverage：被分到具体队列而不是人工复核的比例。
- Accepted accuracy：仅在分到具体队列的子集上计算准确率。
- Review challenge recall：应人工复核的挑战样本被正确转人工的比例。

数据少、风格相似，即便指标较好也不能当作真实客服成绩。评估会保留错误和复核案例，方便查明问题。

## 发布与介绍

GitHub 仓库：[zhaoyilii/support-triage-ai](https://github.com/zhaoyilii/support-triage-ai)。可以按 README 克隆，再执行安装与运行步骤。

仓库简介：

> An explainable ML assistant for customer support triage and human-reviewed response drafts.

推荐 topics：`machine-learning`、`nlp`、`scikit-learn`、`streamlit`、`customer-support`。

项目说明见 `docs/PROJECT_BRIEF.md`。对外介绍时可以强调解决的问题、模型选择和局限，不需要声称它已有真实用户或长期运营历史。理解并亲自运行代码后，再按自己的实际贡献描述开发经历；如被问及，如实说明 AI 辅助开发的部分。

## 可以亲自做的小改进

增加十条不同写法的训练消息、修改回复模板、观察新的混合意图如何被处理，都是适合入门的练习。后续优化使用验证集；频繁查看并针对测试集修改模型后，应准备新的未使用测试集。
