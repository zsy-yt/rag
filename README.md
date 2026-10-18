📖 项目简介
本项目基于 Kaggle 平台完成，构建了一个面向企业内部知识库的 RAG（Retrieval-Augmented Generation）问答系统。
系统读取员工手册类文档 data.jsonl，经文本切分后用中文向量模型 BAAI/bge-large-zh-v1.5 编码并建立 FAISS 向量索引；
用户提问时检索 Top-3 相关片段，连同问题一起送入本地部署的 Qwen2-7B-Instruct 生成回答。
Prompt 中强制约束「仅依据检索上下文作答，无相关信息则回答『知识库中没有相关内容』」，以抑制模型幻觉。
完整代码、索引构建过程与 8 个测试用例见 notebook。

📂 文件说明
ai-interviewer.ipynb：Jupyter Notebook，包含依赖安装、文档加载与切分、向量库构建、RAG Chain 组装与测试（文件名沿用仓库现有 notebook，建议按项目主题重命名）

data.jsonl：知识库原始数据，每行一个 JSON 对象，正文存放在 text 字段

faiss_doctor_index/：运行后自动生成的 FAISS 向量索引目录（保存于 /kaggle/working/）

📌 Notebook 在线查看链接
👉 点击查看完整项目rag/chatbot-rag-with-qwen2-7b.ipynb at main · zsy-yt/rag

⚙️ 环境依赖
python
Python 3.10
torch
transformers
accelerate
sentence-transformers
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
langchainhub
faiss-gpu
建议在 GPU 环境运行（notebook 使用 Kaggle T4）。生成模型 Qwen2-7B-Instruct 与向量模型 BAAI/bge-large-zh-v1.5 均可从 Hugging Face / Kaggle Models 获取。
notebook 中虽安装了 openai、aiohttp、jq，但实际流程未使用，可忽略。

✨ 项目亮点
全本地开源模型栈，无需付费 API：向量化使用中文优化的 BAAI/bge-large-zh-v1.5，生成使用 Qwen2-7B-Instruct（float16 + device_map="auto" 加载），适合对数据隐私敏感的企业内部知识库场景。

完整 RAG 流水线：JSONL 加载 → RecursiveCharacterTextSplitter 切分（chunk=512 / overlap=64）→ bge 向量化 → FAISS 索引持久化 → Top-3 检索 → Prompt 组装 → LLM 生成，各环节均为可替换的模块化组件。

显式抗幻觉约束：Prompt 明确限定「只能使用检索到的上下文，无相关信息直接回答『知识库中没有相关内容』，严禁编造」，并用 3 个知识库外的超纲问题验证拒答行为，全部正确返回「知识库中没有相关内容」。

基于 LangChain LCEL 的链式组装：使用 {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser() 声明式构建问答链，代码简洁、易扩展。

🧠 工作流程
text
data.jsonl → Document 加载 → 文本切分(512/64)
          → bge-large-zh 向量化 → FAISS 索引(保存至本地)
          → 用户提问 → Top-3 检索 → Prompt(上下文+问题)
          → Qwen2-7B-Instruct 生成 → 输出回答
📊 数据集说明
数据集为一份虚构企业「星际桥公司」的员工手册类文档，以 JSONL 格式存储：

每行一个 JSON 对象，text 字段为正文内容

经 512 字符切分后共得到 11 个文档块，属于小规模知识库，用于验证 RAG 链路可行性

内容涵盖公司高管信息、星际航线票价、月球/火星基地概况、员工薪酬职级、福利与行为规范等

🖥️ 测试结果
使用 8 个问题对系统进行验证，结果如下：

#	问题	系统回答	结果
1	星际桥公司的 CEO 是谁？	陈星桥，创始人，前 NASA 工程师	✅ 命中
2	去火星的单程经济舱票价？	120 万地球币	✅ 命中
3	月球基地常驻人数？	600 人（静海区）	✅ 命中
4	火星基地特色餐？	红土土豆泥、火星蓝莓	✅ 命中
5	员工每年免费地球-火星往返额度？	1 次（仅限 P4 及以上）	✅ 命中
6	在静默日说话会有什么惩罚？	知识库中没有相关内容	✅ 正确拒答
7	星际桥公司有多少女员工？	知识库中没有相关内容	✅ 正确拒答
8	星际桥公司去年营收是多少？	知识库中没有相关内容	✅ 正确拒答
5 个知识库内问题全部答对，3 个知识库外问题全部正确拒答，说明检索与抗幻觉约束均生效。

⚠️ 已知限制与改进方向
向量归一化：当前 encode_kwargs={'normalize_embeddings': False}，而 bge 系列模型官方推荐归一化后配合余弦相似度使用，后续将开启归一化并切换 FAISS 的相似度度量。

缺少检索评估：目前仅靠端到端问答结果间接判断，后续可补充 Hit Rate、MRR 等检索指标，量化切分粒度与 Top-K 的影响。

未接入重排序：可直接在检索后加入 bge-reranker 做二阶段精排，提升小知识库下的相关性判断。

文档规模小：11 个文档块仅能验证链路可行性，后续将扩充至更大规模企业文档以测试真实场景表现。

📝 使用说明
下载仓库代码，并将 data.jsonl 上传至 Kaggle 数据集（notebook 中路径为 /kaggle/input/datasets/ysqsww/ymbbkp/data.jsonl）

在 Kaggle 挂载 Qwen2-7B-Instruct 模型（notebook 中路径为 /kaggle/input/qwen2/transformers/qwen2-7b-instruct/1），并开启 GPU

逐块运行 notebook：安装依赖 → 加载数据与切分 → 加载 bge 向量模型 → 构建并保存 FAISS 索引 → 加载 Qwen2 → 组装 RAG Chain

在最后的测试单元中修改 question 变量即可提问，也可复用 qa_chain.invoke(...) 交互式调用

