# ViruFunc-FM 数据分层设计

这份说明把你定义的 ViruFunc-FM 任务，先映射成一个可以真正落地的数据层设计。目标不是“把能下的数据都下下来”，而是让每一层都知道自己为哪类证据负责。

## 四层对象

### 1. 蛋白层

主要服务于 `s_i` 与部分 `x_i`。

- `NCBI RefSeq viral proteins`
  - 作用：病毒 ORF 蛋白序列主入口
  - 价值：覆盖面广，便于和病毒基因组层对齐
- `UniProt reviewed viral annotations`
  - 作用：高质量功能、GO、EC、InterPro/Pfam 交叉引用
  - 价值：作为弱监督与知识蒸馏信号

建议后续补充：

- AlphaFold/ESMFold 结构预测缓存或结构 token
- motif/domain 命中结果
- 结构检索索引（如 Foldseek 结果表）

### 2. 基因组层

主要服务于 `g_i`。

- `NCBI RefSeq viral genomes`
  - 作用：病毒全基因组或片段序列
- `NCBI RefSeq viral GenBank flatfiles`
  - 作用：ORF 坐标、顺反链、基因名、产品名、segment 信息

这一层后面应解析出：

- ORF 顺序
- 重叠关系
- 局部邻域窗口
- segment / scaffold / contig 组织
- mature peptide / polyprotein 派生蛋白

### 3. 样本层

主要服务于 `m_i` 与部分 `e_i`。

- `Virus-Host DB host pairs`
  - 作用：病毒到宿主的显式关联
- `Virus-Host DB lineage table`
  - 作用：补充宿主与病毒 taxonomic lineage
- `NCBI taxonomy taxdump`
  - 作用：taxonomy 标准化与 lineage 展开

建议后续补充：

- 项目来源、生态位、组织、组装质量、样本类型
- host-confidence 分数
- contamination / EVE 风险标签

### 4. 知识层

主要服务于 `Y_i`、`R_i`、`C_i`。

- `GO basic ontology`
  - 作用：通用功能本体
- `InterPro2GO mapping`
  - 作用：结构域到 GO 的桥接

建议后续补充：

- 自建病毒机制本体
- evidence code 词表
- 文献证据表
- anti-CRISPR / anti-CBASS / immune antagonism 专题库

## 数据目录约定

- `data/raw/`
  - 只存原始下载文件
  - 不覆盖，不手改
- `data/interim/`
  - 解压、标准化中间产物
  - 可按数据集 ID 分目录
- `data/processed/`
  - 训练表、检索索引、标签映射表
- `data/provenance/`
  - 下载报告、哈希、盘点信息

## 当前 processed 产物

- `refseq/viral_proteins.tsv.gz`
  - 一行一个蛋白 FASTA 记录
  - 包含 accession、description、organism、length、sha256、sequence
- `refseq/viral_genome_sequences.tsv.gz`
  - 一行一个基因组 FASTA 记录
  - 保留 version accession、description、organism 和序列
- `refseq/viral_genomes.tsv.gz`
  - 一行一个 GenBank 基因组记录
  - 包含 taxid、segment、source host、isolate、geo 信息
- `refseq/viral_cds.tsv.gz`
  - 一行一个带 `protein_id` 的蛋白 feature
  - 既包含 `CDS`，也包含 `mat_peptide` 这类成熟肽 feature
- `taxonomy/observed_taxonomy.tsv.gz`
  - 当前数据集中实际出现 taxid 的标准化 lineage 表
- `taxonomy/virus_host_pairs.standardized.tsv.gz`
  - Virus-Host DB 的标准化主表
- `taxonomy/virus_host_refseq_links.tsv.gz`
  - 按 RefSeq accession 展开的 host link 表，便于和 genome join
- `training/viral_protein_training_index.tsv.gz`
  - 当前蛋白级训练主索引
- `training/viral_genome_training_index.tsv.gz`
  - 当前基因组级训练主索引

## 当前 manifest 分组

- `mvp_core`
  - 立刻下载，支撑最小可运行版本
- `optional_large`
  - 体积大，但后续很可能接入

## 为什么先不自动下载一切

- 某些结构和 domain 库体量极大，MVP 阶段先把高价值核心源稳定下来更划算。
- 后续你可能会根据服务器磁盘、GPU 预算和实验轴选择不同扩展包。
- 先做 manifest 驱动，后面扩展数据源时可以只加配置，不改主逻辑。
