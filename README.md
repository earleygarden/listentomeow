# 听喵说

传一段猫叫，得到一句它「可能」在说的话。**音频全程在本机处理，不上传任何服务器。**

<img src="web/assets/hero.jpg" alt="听喵说" width="100%">

---

## 先说实话：它不是真的在翻译猫语

猫语没有公认的语法，也没有可对照的语料，任何声称「逐句翻译」的产品都在编。

听喵说做的是三件能被验证的事：

1. **确认这是喵叫**（而不是狗吠、人声、哈气）——准确率 93%
2. **猜它属于哪一类叫**（满足 / 索求 / 不安）——三分类 62%
3. **按这一类，配一句合适的中文**——语料是写好的，不是现编的

中间没有语义层。所以结果页永远挂着一句「趣味解读 · 仅供娱乐，不构成任何诊断」。这句话不是免责条款，是这个产品对自己的定义。

**它真正诚实的地方在于：不知道的时候会说不知道。** 如果这段声音离训练分布太远（多猫合唱、嘈杂环境、罕见叫法），它会标「拿不准」并给出「最像 + 备选」，而不是硬猜一个高置信度。

---

## 它怎么工作

```
音频/视频
   ↓ 统一 16kHz 单声道，去首尾静音，最长取 30 秒
① 闸门     SVM → P(猫叫) ≥ 0.54 才放行
   ↓       拦截：狗吠、人声、哈气、过于嘈杂
② 听情绪   AST（AudioSet 预训练）→ 768 维向量 → SVM 三分类
   ↓       满足·放松 / 期待·索求 / 不安·焦虑
③ 挑台词   情绪定语境 → 叠加时间档、声音形状 → 从语料库取句
   ↓       早上七点和深夜十二点，同一段猫叫说的话不一样
④ 护栏     kNN 距离超出训练分布 → 标记「拿不准」，给最像 + 备选
   ↓
一句中文
```

视频额外走一路 `vision.py`（DETR 检测画面里有没有猫）。视频没有音轨时，降级为只报画面（`status: vision_only`）。

---

## 快速开始

### 1. 装依赖

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

还需要本机有 Node.js（服务端用它，无第三方 npm 依赖）。

### 2. 模型文件

仓库里已包含可以直接用的分类器权重：

| 文件 | 作用 | 大小 |
|---|---|---|
| `models/cat_gate.joblib` | 喵叫闸门 | 0.7 MB |
| `models/cat_emotion_model_ast.joblib` | 情绪分类（主） | 2.3 MB |
| `models/cat_emotion_model.joblib` | 情绪分类（MFCC 兜底） | 0.8 MB |
| `models/detr_cat_quant.onnx` | 画面猫检测 | 41 MB |
| `models/ast_feat/*.npy` | AST 预处理参数 | 260 KB |

**唯一需要你自己生成的是 `models/ast_embed.onnx`**（345 MB，超过 GitHub 单文件上限）：

```bash
pip install torch transformers
python -c "from transformers import ASTForAudioClassification as A; A.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593').save_pretrained('data/_ast_torch')"
python scripts/export_ast_onnx.py
```

> 不生成也能跑。`predict.py` 会自动降级到 MFCC 模型，准确率从 62% 降到 56–58%，功能完整、只是没那么准。

### 3. 启动

```bash
run.bat                # Windows：起服务 + 自动开浏览器
node server.js         # 或手动起，然后访问 http://127.0.0.1:8790
```

命令行直接听一段：

```bash
python scripts/predict.py samples/sample_饿_等食.wav
```

---

## 实测指标

所有数字都来自按「猫 ID」分组的交叉验证（GroupKFold），避免同一只猫的数据同时出现在训练和测试里。

**闸门**（878 段 / 133 只猫，猫叫 637 vs 非猫叫 241）

- 交叉验证准确率 **93.17%**，AUC 0.973，判定阈值 0.54
- 狗吠 440 段：拦截率 100%
- 训练集喵叫：误拦率 2.7%
- 猫的其它发声（哈气、呼噜）：拦截率 30.8%

**情绪三分类**（440 段，3 类，随机基线 33%）

- AST 特征 + SVM：**62.0%**
- MFCC 特征 + SVM：56–58%
- 分类型看：「不安」最稳（precision 0.78），「满足」和「期待」互相误判最多——恰好是日常最常录的两类

---

## 已知局限

- **慢**：每次请求新起一个 Python 进程，345 MB 的 AST 模型要重新加载，首次约 13 秒。这是当前最该修的问题。
- **语料库还薄**：约 47 条，设计骨架是 144 条（12 个时间×形状组合 × 6 猫格 × 2 语气）。「换一个说法」按两三次就会重复。
- **三分类太粗**：真实猫叫的意图远不止三类，扩类需要补数据。
- **只认喵叫**：呼噜、哈气、低吼目前归到「不是喵叫」这一档，没有细分。

---

## 目录结构

```
cat-translator/
├─ run.bat                      一键启动
├─ server.js                    本地服务（零 npm 依赖，127.0.0.1:8790）
├─ web/
│  ├─ index.html                单文件前端（上传/录音/实时波形/结果/历史/档案馆）
│  └─ assets/hero.jpg           页头插画
├─ scripts/
│  ├─ features.py               声学特征：MFCC + 谱特征 + f0 → 268 维
│  ├─ ast_features.py           AST 768 维 embedding（onnxruntime，无需 torch）
│  ├─ quote_bank.py             「猫的嘴替」语料库与挑句逻辑
│  ├─ predict.py                听到什么：闸门 → 情绪 → 出话
│  ├─ vision.py                 看到什么：DETR 检测画面里有没有猫
│  ├─ train.py                  情绪模型（MFCC 版）
│  ├─ train_ast.py              情绪模型（AST 版）
│  ├─ train_gate.py             喵叫闸门
│  └─ export_ast_onnx.py        导出 AST 骨干为 ONNX
├─ models/                      分类器权重 + 猫检测模型
├─ samples/                     4 段示例音频
└─ docs/语料样本_v1_猫的嘴替.md   语料库设计骨架与写句规矩
```

---

## 隐私

- 音频只在内存和临时文件里过一遍，处理完立即删除
- 分析存档（原始录音 + 结果）只落在本机 `data/uploads/`，**已在 `.gitignore` 中排除**
- 没有任何埋点、没有云端调用、不需要登录

---

## 数据来源与使用限制

- 训练数据：**CatMeows**（Ludovico et al., 2020, Zenodo, DOI `10.5281/zenodo.4008297`），许可为**科研与非商业用途**
- 负样本：**BarkMeowDB**
- AST 骨干：`MIT/ast-finetuned-audioset-10-10-0.4593`
- 猫检测：DETR-ResNet-50（COCO 类别 id 17）

**因此：本项目的模型权重仅限研究与个人使用，不得商用。** 若要商用，需替换为自有采集数据或取得授权。代码部分暂未声明开源许可。

---

## 作者

鳄梨导演 · 一个人 + 一套 AI 工具做出来的东西。
