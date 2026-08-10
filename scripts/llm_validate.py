"""NLP 伪标签 — 第一阶段验证脚本.

在 58 个有标签的放射报告上测试 LLM 提取 12 类膝关节异常的能力。
支持任何 OpenAI-compatible API (DeepSeek, GPT-4, Claude via proxy, 本地 vLLM 等).

输出:
    data/pseudo_labels_valid.csv     逐样本对比 (预测 vs 真值)
    data/nlp_validation_report.csv   Per-class 指标
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 从同级目录的 api_config.py 读取 API 配置
# 首次使用: cp scripts/api_config.example.py scripts/api_config.py 然后填入 key
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from api_config import API_KEY, API_BASE, MODEL as DEFAULT_MODEL
except ImportError:
    print("[ERROR] 找不到 scripts/api_config.py")
    print("  1. cp scripts/api_config.example.py scripts/api_config.py")
    print("  2. 编辑 api_config.py, 填入你的 API key")
    sys.exit(1)

LIMIT = 0             # 跑前 N 个样本 (0 = 全部 58 个)

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── 标签列名 ──────────────────────────────────────────────────
LABEL_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# ── 已报告中这 12 类的常见表达 (多语言关键词映射) ──────────────
_CLASS_HINTS = {
    "ACL": "前交叉韧带 / LCA / anterior cruciate ligament",
    "MCL": "内侧副韧带 / LCM / medial collateral ligament",
    "Medial Meniscus": "内侧半月板 / menisco medial/interno / medial meniscus",
    "Lateral Meniscus": "外侧半月板 / menisco lateral/externo / lateral meniscus",
    "Medial OA": "内侧骨关节炎 / artrosis medial / medial osteoarthritis / medial chondrosis",
    "Lateral OA": "外侧骨关节炎 / artrosis lateral / lateral osteoarthritis / lateral chondrosis",
    "PF OA": "髌股骨关节炎 / artrosis patelofemoral / patellofemoral osteoarthritis / trochlear cartilage defect",
    "Effusion": "关节积液 / derrame articular / joint effusion / fluid collection",
    "Synovitis": "滑膜炎 / sinovitis / synovitis / synovial thickening",
    "Baker's": "腘窝囊肿 / quiste poplíteo / Baker's cyst / popliteal cyst",
    "Contusion": "骨挫伤 / contusión ósea / bone bruise / bone edema / marrow edema",
    "Fracture": "骨折 / fractura / fracture",
}

# ── 置信度规则: 每类的高/低置信关键词 ─────────────────────────
# high: 报告中出现这些词 → LLM 预测为 1 时置信度高
# low:  报告中只出现这些词(没有 high) → LLM 预测为 1 时可能是过度解读
CONFIDENCE_RULES: dict[str, dict[str, list[str]]] = {
    "ACL": {
        "high": [
            # 英文 — 常见词序
            "acl tear", "acl rupture", "torn acl", "acl graft tear",
            "tear of the acl", "tear of acl", "tear of the anterior cruciate",
            "rupture of the acl", "rupture of acl", "rupture of the anterior cruciate",
            "complete tear of the acl", "complete tear of the anterior cruciate",
            "acl is torn", "acl completely torn", "acl disruption",
            "anterior cruciate ligament tear", "anterior cruciate ligament rupture",
            "acl deficient", "absent acl", "acl transection",
            "complete acl", "full thickness acl", "acl avulsion",
            "acute acl", "chronic acl", "acl sprain", "acl injury",
            # 西班牙语
            "rotura del lca", "lca roto", "lca rupture",
            "lesión del lca", "desgarro del lca",
            "rotura de lca", "rotura completa del lca",
            # 西班牙语 — "Rotura ... del LCA" (中间有词)
            "proximal del lca", "tercio del lca", "del lca asociado",
            "del lca con", "del lca.",
            # 土耳其语
            "ön çapraz bağ yırtığı", "ön çapraz bağ rüptürü",
            "ön çapraz bağ kopması",
            # 荷兰语
            "voorste kruisband ruptuur", "voorste kruisband scheur",
            "voorste kruisband letsel",
            # 希腊语
            "ρήξη πρόσθιου χιαστού", "πρόσθιου χιαστού ρήξη",
            "πρόσθιος χιαστός ρήξη",
        ],
        "low": [
            "lca", "acl", "anterior cruciate", "ön çapraz bağ",
            "voorste kruisband", "πρόσθιου χιαστού",
        ],
    },
    "MCL": {
        "high": [
            # 英文
            "mcl tear", "mcl rupture", "mcl sprain", "mcl injury",
            "tear of the mcl", "rupture of the mcl", "injury of the mcl",
            "tear of the medial collateral", "rupture of the medial collateral",
            "complete rupture of the mcl", "complete tear of the mcl",
            "mcl grade", "mcl partial", "mcl complete",
            "medial collateral ligament tear", "medial collateral ligament rupture",
            "medial collateral ligament sprain", "medial collateral ligament injury",
            # 西班牙语
            "rotura del lcm", "lcm roto",
            "rotura parcial del lcm", "rotura del ligamento colateral medial",
            "lesión del lcm", "esguince del ligamento colateral medial",
            "rotura de lcm",
            # 荷兰语
            "mediale collaterale band", "mediale collateraal ligament",
            # 希腊语
            "έσω πλάγιου συνδέσμου ρήξη", "έσω πλάγιος σύνδεσμος",
        ],
        "low": [
            "mcl", "lcm", "medial collateral", "collateral ligament",
            "ligamento colateral medial",
        ],
    },
    "Medial Meniscus": {
        "high": [
            "medial meniscus tear", "medial meniscal tear",
            "tear of the medial meniscus", "tear of medial meniscus",
            "tearing of the medial", "tearing of medial",
            "medial meniscus: tear", "medial meniscus: extensive",
            "medial meniscus: complete", "medial meniscus: complex",
            "medial meniscus: horizontal", "medial meniscus: radial",
            "medial meniscus: vertical", "medial meniscus: bucket",
            "medial meniscus: flap", "medial meniscus: maceration",
            "medial meniscus radial", "medial meniscus horizontal",
            "medial meniscus vertical", "medial meniscus complex",
            "medial meniscus bucket", "medial meniscus flap",
            "medial meniscus maceration", "medial meniscal rupture",
            "medial meniscal degeneration", "medial meniscus degeneration",
            "degenerative tear of medial",
            # 西班牙语
            "rotura de menisco interno", "rotura del menisco medial",
            "menisco interno roto", "rotura menisco interno",
            "desgarro de menisco interno", "lesión del menisco interno",
            "rotura del cuerno posterior del menisco interno",
            # 荷兰语
            "mediale meniscus scheur", "mediale meniscus ruptuur",
            "scheur van de mediale meniscus",
            # 希腊语
            "έσω μηνίσκου ρήξη", "έσω μηνίσκος ρήξη",
            "ρήξη έσω μηνίσκου", "ρήξη του έσω μηνίσκου",
        ],
        "low": [
            "medial meniscus", "menisco interno", "menisco medial",
        ],
    },
    "Lateral Meniscus": {
        "high": [
            "lateral meniscus tear", "lateral meniscal tear",
            "tear of the lateral meniscus", "tear of lateral meniscus",
            "tearing of the lateral", "tearing of lateral",
            "lateral meniscus: tear", "lateral meniscus: extensive",
            "lateral meniscus: complete", "lateral meniscus: complex",
            "lateral meniscus: horizontal", "lateral meniscus: radial",
            "lateral meniscus: vertical", "lateral meniscus: bucket",
            "lateral meniscus: flap", "lateral meniscus: maceration",
            "lateral meniscus radial", "lateral meniscus horizontal",
            "lateral meniscus vertical", "lateral meniscus complex",
            "lateral meniscus bucket", "lateral meniscus flap",
            "lateral meniscal rupture",
            # 西班牙语
            "rotura de menisco externo", "rotura del menisco lateral",
            "menisco externo roto", "rotura menisco externo",
            "desgarro de menisco externo", "lesión del menisco externo",
            # 荷兰语
            "laterale meniscus scheur", "laterale meniscus ruptuur",
            # 希腊语
            "έξω μηνίσκου ρήξη", "έξω μηνίσκος ρήξη",
            "ρήξη έξω μηνίσκου", "ρήξη του έξω μηνίσκου",
        ],
        "low": [
            "lateral meniscus", "menisco externo", "menisco lateral",
        ],
    },
    "Medial OA": {
        "high": [
            # 英文
            "medial osteoarthritis", "medial compartment osteoarthritis",
            "medial chondrosis", "medial compartment chondrosis",
            "medial oa", "medial cartilage loss",
            "medial compartment cartilage loss", "medial compartment narrowing",
            "medial chondropathy", "medial compartment chondropathy",
            "medial compartment cartilage", "medial compartment degeneration",
            "medial tibiofemoral osteoarthritis", "medial tibiofemoral chondrosis",
            "osteoarthritis of the medial", "chondrosis of the medial",
            "cartilage loss in the medial", "cartilage loss medial",
            "medial degenerative changes", "medial degenerative joint",
            # "OA of all three" 应该触发所有 OA
            "oa of all three", "oa of all 3",
            "osteoarthritis of all three", "osteoarthritis of all compartments",
            "tricompartmental osteoarthritis", "tricompartmental chondrosis",
            "incipient oa of all",
            # 西班牙语
            "artrosis medial", "artrosis de compartimento medial",
            "artrosis femorotibial medial", "condropatía medial",
            "artrosis fémoro-tibial medial",
            # 荷兰语
            "mediaal kraakbeenlijden", "mediaal kraakbeenverlies",
            "mediale kraakbeen", "mediale artrose",
            "kraakbeenverlies mediaal", "kraakbeenlijden mediaal",
            # 希腊语
            "έσω διαμερίσματος οστεοαρθρίτιδα",
            # 保加利亚语
            "медиална остеоартроза", "медиална хондроза",
        ],
        "low": [
            "chondrosis", "osteoarthritis", "cartilage loss",
            "artrosis", "condropatía", "kraakbeenlijden",
            "kraakbeenverlies", "остеоартроза",
        ],
    },
    "Lateral OA": {
        "high": [
            # 英文
            "lateral osteoarthritis", "lateral compartment osteoarthritis",
            "lateral chondrosis", "lateral oa", "lateral cartilage loss",
            "lateral compartment narrowing", "lateral chondropathy",
            "lateral compartment chondrosis", "lateral compartment cartilage loss",
            "lateral tibiofemoral osteoarthritis", "lateral tibiofemoral chondrosis",
            "osteoarthritis of the lateral", "chondrosis of the lateral",
            "cartilage loss in the lateral", "cartilage loss lateral",
            "lateral degenerative changes", "lateral degenerative joint",
            # "OA of all three" variants
            "oa of all three", "oa of all 3",
            "osteoarthritis of all three", "osteoarthritis of all compartments",
            "tricompartmental osteoarthritis", "tricompartmental chondrosis",
            "incipient oa of all",
            # 西班牙语
            "artrosis lateral", "artrosis de compartimento lateral",
            "artrosis femorotibial lateral", "condropatía lateral",
            "artrosis fémoro-tibial lateral",
            # 荷兰语
            "lateraal kraakbeenlijden", "lateraal kraakbeenverlies",
            "laterale kraakbeen", "laterale artrose",
            "kraakbeenverlies lateraal",
            "lateraal femorotibiaal kraakbeenlijden",
            "lateraal femorotibiaal kraakbeenverlies",
            # 希腊语
            "έξω διαμερίσματος οστεοαρθρίτιδα",
        ],
        "low": [
            "chondrosis", "osteoarthritis", "cartilage loss", "artrosis",
        ],
    },
    "PF OA": {
        "high": [
            "patellofemoral osteoarthritis", "patellofemoral chondrosis",
            "pf oa", "artrosis patelofemoral", "trochlear cartilage defect",
            "patellar chondrosis", "retropatellar chondrosis",
            "retropatellar cartilage loss", "patellofemoral cartilage loss",
            "condropatía rotuliana", "condropatía patelofemoral",
        ],
        "low": [
            "patellofemoral", "patellar", "trochlear", "rotuliana",
        ],
    },
    "Effusion": {
        "high": [
            "effusion", "joint effusion", "knee effusion",
            "derrame", "derrame articular",
            "large effusion", "moderate effusion", "small effusion",
            "suprapatellar effusion", "effusion is present",
            "effusion present", "joint fluid", "intra-articular fluid",
        ],
        "low": [
            "fluid", "edema", "swelling", "líquido", "sıvı",
        ],
    },
    "Synovitis": {
        "high": [
            "synovitis", "sinovitis", "synovial thickening",
            "synovial inflammation", "synovial proliferation",
            "hoffitis", "hoffa", "hoffa's", "hoffa's fat pad",
            "engrosamiento sinovial", "sinovial thickening",
            "fat pad stranding", "fat pad edema",
            "síndrome de pinzamiento de la almohadilla grasa",
        ],
        "low": [
            "synovial", "synovium", "sinovial",
        ],
    },
    "Baker's": {
        "high": [
            "baker's cyst", "baker cyst", "popliteal cyst",
            "baker's", "bakers cyst",
            "quiste poplíteo", "quiste popliteo", "quiste de baker",
            "popliteal cyst", "popliteal cyst",
        ],
        "low": [
            "cyst", "popliteal", "quiste",
        ],
    },
    "Contusion": {
        "high": [
            "bone contusion", "bone bruise", "bone bruising",
            "contusión ósea", "contusiones óseas",
            "hematoma óseo", "trabecular microfracture",
            "контузия", "контузионен", "contusión", "contusiones",
            "contusio", "bone contusions",
        ],
        "low": [
            "bone edema", "marrow edema", "bone marrow edema",
            "edema óseo", "edema oseo", "osseous edema",
            "bone marrow oedema", "oedema",
            "kemik iliği ödemi", "костномозъчен едем",
            "edema", "bone marrow", "marrow",
        ],
    },
    "Fracture": {
        "high": [
            "fracture", "fractura", "fracture line", "fractured",
            "split fracture", "tibial plateau fracture",
            "insufficiency fracture", "subchondral fracture",
            "depression fracture", "avulsion fracture",
            "broken", "fracturas", "фрактура",
        ],
        "low": [
            "trauma", "acute injury",
        ],
    },
}

# ── 强否定模式: 报告中有这些词 → 强制翻转为 0 ─────────────────
# 用于规则修正层: LLM=1 但报告明确否定 → 纠正为 0
STRONG_NEGATIONS: dict[str, list[str]] = {
    "ACL": [
        "acl is intact", "acl intact", "acl appears normal",
        "acl is normal", "acl normal", "anterior cruciate ligament is intact",
        "lca intacto", "lca normal", "lca está intacto",
        "sin rotura del lca", "no acl tear", "no evidence of acl tear",
    ],
    "MCL": [
        "mcl is intact", "mcl intact", "mcl appears normal",
        "mcl normal", "medial collateral ligament is intact",
        "lcm intacto", "lcm normal", "sin rotura del lcm",
        "no mcl tear",
    ],
    "Medial Meniscus": [
        "medial meniscus is intact", "medial meniscus intact",
        "medial meniscus is normal", "no medial meniscal tear",
        "no tear of the medial meniscus", "medial meniscus not torn",
        "menisco interno intacto", "menisco interno normal",
        "sin rotura del menisco interno", "no medial meniscus tear",
    ],
    "Lateral Meniscus": [
        "lateral meniscus is intact", "lateral meniscus intact",
        "lateral meniscus is normal", "no lateral meniscal tear",
        "no tear of the lateral meniscus", "lateral meniscus not torn",
        "menisco externo intacto", "menisco externo normal",
        "sin rotura del menisco externo", "no lateral meniscus tear",
    ],
    "Medial OA": [
        "no medial osteoarthritis", "no medial chondrosis",
        "no medial cartilage loss", "medial compartment is normal",
        "sin artrosis medial", "sin condropatía medial",
        "no evidence of medial oa",
    ],
    "Lateral OA": [
        "no lateral osteoarthritis", "no lateral chondrosis",
        "no lateral cartilage loss", "lateral compartment is normal",
        "sin artrosis lateral", "sin condropatía lateral",
        "no evidence of lateral oa",
    ],
    "PF OA": [
        "no patellofemoral osteoarthritis", "no pf oa",
        "no patellofemoral chondrosis", "patellofemoral joint is normal",
        "sin artrosis patelofemoral", "no pf chondrosis",
        "trochlear cartilage is normal", "patellar cartilage is normal",
    ],
    "Effusion": [
        "no effusion", "no joint effusion", "no knee effusion",
        "sin derrame", "sin derrame articular",
        "effusion is absent", "no significant effusion",
        "no pathological effusion",
    ],
    "Synovitis": [
        "no synovitis", "no sinovitis", "no synovial thickening",
        "no synovial inflammation", "sin sinovitis",
        "synovium is normal", "no hoffitis",
    ],
    "Baker's": [
        "no baker's cyst", "no baker cyst", "no popliteal cyst",
        "sin quiste poplíteo", "sin quiste de baker",
        "baker's cyst is not", "popliteal cyst is not",
    ],
    "Contusion": [
        "no contusion", "no bone contusion", "no bone bruise",
        "sin contusión", "sin contusión ósea", "no bone bruising",
        "no evidence of contusion",
    ],
    "Fracture": [
        "no fracture", "no fractura", "sin fractura",
        "fracture is not seen", "no evidence of fracture",
        "no acute fracture", "no displaced fracture",
        "no fracture line", "fractures are not",
    ],
}

# ── System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior musculoskeletal radiologist. Extract 12 binary knee MRI findings from radiology reports.

## OUTPUT FORMAT
Return ONLY a JSON object. No markdown, no explanation, no code fences.
{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker's":0,"Contusion":0,"Fracture":0}

## CLASS DEFINITIONS & DECISION RULES

**ACL** (anterior cruciate ligament)
  1 if: tear, rupture, complete disruption, "no intact fibers", avulsion fracture of ACL attachment
  0 if: intact, normal, sprain-only without tear, "mucoid degeneration", "ganglion cyst of ACL", "cystic transformation"
  KEYWORDS for 1: tear of ACL, ACL rupture, torn ACL, rotura del LCA, ACL deficient, ACL graft tear
  KEYWORDS for 0: ACL intact, ACL normal, ACL sprain (grade 1), mucoid ACL, cystic ACL

**MCL** (medial collateral ligament)
  1 if: tear, rupture, grade 2-3 sprain, complete disruption
  0 if: intact, normal, grade 1 sprain, "thickening without tear"
  Note: "sprain" alone without grade → 0. "Grade 2/3 sprain" or "partial/full tear" → 1.

**Medial Meniscus**
  1 if: tear, rupture, maceration, degenerative tear, complete radial/flap/bucket-handle tear
  0 if: intact, normal, "no tear", "degeneration without tear", "intrasubstance degeneration"

**Lateral Meniscus**
  1 if: same criteria as medial meniscus, applied to lateral compartment

**Medial OA** (medial compartment osteoarthritis)
  1 if: medial chondrosis, medial cartilage loss, medial osteoarthritis, medial joint space narrowing, "OA of medial compartment", "tricompartmental OA"
  0 if: isolated patellofemoral OA, lateral-only OA, "physiological cartilage thinning"

**Lateral OA** (lateral compartment osteoarthritis)
  1 if: lateral chondrosis, lateral cartilage loss, lateral osteoarthritis, lateral joint space narrowing, "tricompartmental OA"
  0 if: same logic as medial OA

**PF OA** (patellofemoral osteoarthritis)
  1 if: patellofemoral chondrosis, trochlear cartilage defect, patellar chondrosis, retropatellar cartilage loss, "OA of patellofemoral joint"
  0 if: "mild chondral thinning" without full-thickness defect in young patient

**Effusion** (joint effusion)
  1 if: "joint effusion", "effusion", "derrame articular", described as moderate/large/substantial
  0 if: "trace fluid", "physiological fluid", "minimal fluid", "small amount of fluid" without pathology
  CRITICAL: "soft tissue edema", "bone marrow edema", "subcutaneous edema" are NOT effusion!
  CRITICAL: "High signal" without "effusion" or "fluid" → 0.

**Synovitis**
  1 if: "synovitis", "sinovitis", "synovial thickening", "synovial proliferation", "synovial inflammation",
       "hoffitis", "Hoffa's fat pad edema/impingement/stranding", "fat pad stranding"
  0 if: no mention of synovial abnormality
  CRITICAL: "hoffitis" or "Hoffa's fat pad" with abnormality → Synovitis=1.

**Baker's** (Baker's cyst / popliteal cyst)
  1 if: "Baker's cyst", "popliteal cyst", "quiste poplíteo/de Baker"
  0 if: "no Baker's cyst", "no popliteal cyst", or not mentioned

**Contusion** (bone contusion / bone bruise)
  1 if: "bone contusion", "bone bruise", "contusión ósea", "trabecular microfracture", "bone bruising"
  0 if: ONLY "bone marrow edema", "marrow edema", "bone edema", "edema óseo" — edema alone is NOT contusion!
       Edema can be degenerative, reactive, or atraumatic. Only label contusion if explicitly called a contusion/bruise.
  CRITICAL: This is the #1 source of false positives. Bone marrow edema ≠ contusion unless explicitly labeled as such.

**Fracture**
  1 if: "fracture", "fractura", "fracture line", "broken", "insufficiency fracture", "subchondral fracture"
  0 if: "no fracture", "sin fractura", fracture explicitly excluded
  CRITICAL: Read negation carefully. "No fracture is seen" → 0.

## CRITICAL DISTINCTIONS (common errors)

1. "bone marrow edema" / "bone edema" / "marrow edema" / "edema óseo" → Contusion=0, Effusion=0
   These are signal patterns, NOT diagnoses. Only label Contusion if report says "contusion" or "bruise".

2. "soft tissue edema" / "subcutaneous edema" → NOT Effusion, NOT Contusion
   These are extra-articular. Do NOT label any class based on soft tissue edema.

3. "cystic transformation of ACL" / "mucoid degeneration of ACL" → ACL=0
   Degenerative changes are NOT tears.

4. "chondrosis" without compartment specified → check context carefully
   If report describes generalized chondrosis → check which compartments are affected.

5. "hoffitis" / "Hoffa's fat pad impingement" / "fat pad stranding" → Synovitis=1
   These are forms of synovial inflammation. Do NOT miss them.

6. Negation: "no X", "X is intact", "X is normal", "sin X", "without X" → X=0
   Pay special attention to negation words in all languages.

## MULTILINGUAL VOCABULARY

Spanish:
  rotura/desgarro = tear/rupture | derrame = effusion | contusión = contusion
  menisco interno = medial meniscus | menisco externo = lateral meniscus
  LCA = ACL | LCM = MCL | LCP = PCL (NOT ACL!) | ligamento colateral = collateral ligament
  artrosis = osteoarthritis | condropatía = chondropathy | sinovitis = synovitis
  edema óseo = bone edema (NOT contusion!) | fractura = fracture
  quiste poplíteo/de Baker = Baker's cyst
  "sin hallazgos significativos" = no significant findings

Dutch:
  voorste kruisband = ACL | mediale collaterale band = MCL
  meniscus = meniscus (mediale/laterale) | kraakbeenlijden = chondropathy
  kraakbeenverlies = cartilage loss | scheur/ruptuur = tear/rupture
  gewrichtseffusie = joint effusion | botkneuzing = bone bruise

Greek:
  ρήξη = tear/rupture | εκφυλιστική ρήξη = degenerative tear
  πρόσθιος χιαστός = ACL | έσω μηνίσκος = medial meniscus | έξω μηνίσκος = lateral meniscus
  οστεοαρθρίτιδα = osteoarthritis | χονδροπάθεια = chondropathy
  συλλογή υγρού = fluid collection/effusion | κάταγμα = fracture

Turkish:
  ön çapraz bağ = ACL | medial menisküs = medial meniscus
  yırtık = tear | efüzyon = effusion | kırık = fracture
  kemik iliği ödemi = bone marrow edema (NOT necessarily contusion!)

Bulgarian:
  фрактура = fracture | контузия = contusion | разкъсване = tear
  остеоартроза = osteoarthritis | менискус = meniscus
  костномозъчен едем = bone marrow edema (NOT necessarily contusion!)

## DECISION FLOW
For each class, ask yourself:
1. Is there an EXPLICIT statement of this abnormality? → 1
2. Is there an EXPLICIT statement that this structure is NORMAL? → 0
3. Is the structure mentioned but only with degenerative/incidental/non-pathological findings? → 0
4. Is the structure not mentioned at all? → 0
5. Am I confusing a signal pattern (edema, high signal) with a diagnosis? → re-read, likely 0

Output ONLY the JSON object, nothing else:
{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker's":0,"Contusion":0,"Fracture":0}"""

# ── Few-shot 示例: 覆盖 5 种关键模式 ─────────────────────────
# 每条是 (report, expected_json) — 会被插入到 messages 中
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # 例 1: 多发性损伤 (ACL + 半月板 + 积液 + 骨关节炎)
    (
        "MRI RIGHT KNEE: 1. Complete rupture of the ACL at its femoral attachment. "
        "2. Horizontal tear involving the posterior horn and body of the medial meniscus. "
        "3. Moderate-sized knee joint effusion. "
        "4. Full-thickness cartilage loss in the medial compartment consistent with advanced osteoarthritis. "
        "5. The lateral meniscus is intact. The MCL, LCL, and PCL are intact. "
        "No fracture. No bone contusion or bruise.",
        '{"ACL":1,"MCL":0,"Medial Meniscus":1,"Lateral Meniscus":0,"Medial OA":1,"Lateral OA":0,"PF OA":0,"Effusion":1,"Synovitis":0,"Baker\'s":0,"Contusion":0,"Fracture":0}',
    ),
    # 例 2: 骨水肿但非挫伤 — 最常见的假阳性来源!
    (
        "Subchondral bone marrow edema is present in the lateral femoral condyle and medial tibial plateau, "
        "likely representing degenerative or reactive change. No discrete fracture line is identified. "
        "No bone contusion. No bone bruise. "
        "The ACL, PCL, MCL, and LCL appear intact. Both menisci are normal in contour and signal. "
        "No joint effusion. No synovitis. No popliteal cyst.",
        '{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker\'s":0,"Contusion":0,"Fracture":0}',
    ),
    # 例 3: 滑膜炎 + Hoffa 脂肪垫 (hoffitis 容易漏检!)
    (
        "Diffuse synovial thickening and hyperenhancement is present throughout the knee joint, "
        "consistent with active synovitis. There is also soft tissue edema and stranding within "
        "Hoffa's fat pad consistent with hoffitis. "
        "Small popliteal cyst is noted. "
        "No meniscal tear. Cruciate and collateral ligaments are intact. "
        "No fracture. No bone contusion. No significant chondrosis.",
        '{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":1,"Baker\'s":1,"Contusion":0,"Fracture":0}',
    ),
    # 例 4: 西班牙语 — 多发性损伤 + 骨挫伤
    (
        "RMN de rodilla derecha. Hallazgos: "
        "Rotura completa del LCA en su inserción femoral proximal. "
        "Rotura longitudinal del cuerno posterior del menisco medial que se extiende al cuerpo. "
        "Derrame articular de moderada cuantía. "
        "Contusiones óseas en cóndilo femoral lateral y meseta tibial lateral. "
        "El menisco lateral está intacto. Los ligamentos colaterales están intactos. "
        "No se observa fractura.",
        '{"ACL":1,"MCL":0,"Medial Meniscus":1,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":1,"Synovitis":0,"Baker\'s":0,"Contusion":1,"Fracture":0}',
    ),
    # 例 5: 阴性报告 (有退化信号但无病理)
    (
        "MRI knee without contrast. "
        "Minimal physiological fluid in the joint space, no pathological effusion. "
        "Mild intrasubstance degeneration of the medial meniscus without discrete tear. "
        "Mild chondral thinning of the patellofemoral compartment without focal defect. "
        "ACL, PCL, MCL, LCL all intact. No fracture. No bone bruise or contusion. "
        "No synovitis. No Baker's cyst.",
        '{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker\'s":0,"Contusion":0,"Fracture":0}',
    ),
]


def parse_llm_response(text: str) -> dict[str, int] | None:
    """从 LLM 回复中提取 JSON 标签. 容忍各种格式."""
    text = text.strip()

    # 策略 1: 找第一个完整 JSON 对象
    # 允许嵌套但只匹配外层
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    result = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                # 检查是否包含所有 12 个 key
                if all(col in result for col in LABEL_COLS):
                    parsed = {}
                    for col in LABEL_COLS:
                        v = result.get(col)
                        if v is None:
                            break
                        parsed[col] = int(v)
                    else:
                        return parsed

    # 策略 2: 用单引号
    try:
        result = json.loads(text.replace("'", '"'))
        if all(col in result for col in LABEL_COLS):
            return {col: int(result[col]) for col in LABEL_COLS}
    except json.JSONDecodeError:
        pass

    return None


def _is_negated(text: str, keyword: str, window: int = 40) -> bool:
    """检查关键词是否出现在否定语境中.

    在关键词前 window 个字符内搜索否定词.
    窗口较小 (40 chars) 以避免跨结构误判.
    """
    text_lower = text.lower()
    kw_lower = keyword.lower()
    # 先按句子分割，只在同一句子内检查否定
    # 这样可以避免 "meniscus is not torn ... ACL is torn" 的跨结构误判
    sentences = text_lower.replace("\n", ". ").split(". ")
    # 找到包含关键词的句子
    for sent in sentences:
        if kw_lower not in sent:
            continue
        idx = sent.find(kw_lower)
        # 只看关键词前的部分
        prefix = sent[:idx]
        # 只看 window 字符内的否定词
        local_prefix = prefix[-window:] if len(prefix) > window else prefix
        negations = [
            "no ", "not ", "without ", "sin ", "negative for ",
            "absent", "unremarkable", "rules out", "excluded",
            "free of", "no evidence of", "no acute", "no displaced",
            "no focal", "no abnormal", "no significant",
        ]
        if any(n in local_prefix for n in negations):
            return True
    return False


def score_confidence(
    report: str, llm_labels: dict[str, int]
) -> dict[str, str]:
    """基于报告原文中的关键词证据, 对 LLM 的每个预测打分.

    Returns:
        dict: {class: "HIGH"|"MEDIUM"|"LOW"|"REVIEW"}
    """
    report_lower = report.lower()
    confidence: dict[str, str] = {}

    # 通用病变指示词 — 如果报告中有这些词, 说明确实有异常
    PATHOLOGY_INDICATORS = [
        # 韧带/半月板损伤
        "tear", "tearing", "torn", "rupture", "rotura", "desgarro",
        "complete", "extensive", "full thickness", "full-thickness",
        "maceration", "degeneration", "degenerative",
        # OA 相关
        "chondrosis", "osteoarthritis", "cartilage loss", "narrowing",
        "artrosis", "osteofytose", "osteophyte", "kraakbeenlijden",
        "kraakbeenverlies", "condropatía",
        # 积液/滑膜炎
        "effusion", "derrame", "synovitis", "sinovitis",
        # 骨损伤
        "contusion", "contusión", "bruise", "fracture", "fractura",
        "bone marrow edema", "marrow edema", "bone edema",
    ]

    for col in LABEL_COLS:
        rules = CONFIDENCE_RULES.get(col, {})
        high_kw_list: list[str] = rules.get("high", [])
        low_kw_list: list[str] = rules.get("low", [])

        has_high = any(
            kw in report_lower and not _is_negated(report, kw)
            for kw in high_kw_list
        )
        has_low = any(
            kw in report_lower and not _is_negated(report, kw)
            for kw in low_kw_list
        )
        # 报告中是否有任何病变指示词
        has_pathology = any(
            pi in report_lower for pi in PATHOLOGY_INDICATORS
        )

        pred = llm_labels[col]

        if pred == 1:
            if has_high:
                confidence[col] = "HIGH"
            elif has_low and has_pathology:
                # LOW 关键词 + 病变指示词 → 大概率是真阳性, 升级
                confidence[col] = "HIGH"
            elif has_low:
                confidence[col] = "LOW"
            else:
                confidence[col] = "MEDIUM"
        else:  # pred == 0
            if has_high:
                confidence[col] = "REVIEW"
            else:
                confidence[col] = "HIGH"

    return confidence


def apply_rule_correction(
    report: str, llm_labels: dict[str, int]
) -> dict[str, int]:
    """基于关键词规则纠正 LLM 的明显错误.

    两层修正:
      1. 报告有 HIGH 关键词 (未否定) 但 LLM=0 → 纠正为 1 (LLM 漏检)
      2. 报告有强否定模式 但 LLM=1         → 纠正为 0 (LLM 幻觉)
    """
    report_lower = report.lower()
    corrected = dict(llm_labels)
    flips: list[str] = []

    for col in LABEL_COLS:
        rules = CONFIDENCE_RULES.get(col, {})
        high_kw_list: list[str] = rules.get("high", [])
        neg_list: list[str] = STRONG_NEGATIONS.get(col, [])

        # 检查 HIGH 关键词 (未被否定)
        has_high = any(
            kw in report_lower and not _is_negated(report, kw)
            for kw in high_kw_list
        )
        # 检查强否定
        has_strong_negation = any(neg in report_lower for neg in neg_list)

        if has_high and llm_labels[col] == 0:
            corrected[col] = 1
            flips.append(f"  [0→1] {col}: keyword evidence overrides LLM")
        elif has_strong_negation and llm_labels[col] == 1:
            corrected[col] = 0
            flips.append(f"  [1→0] {col}: strong negation overrides LLM")

    if flips:
        pass  # flips are tracked but printed from main loop

    return corrected


def call_llm_with_voting(
    report: str,
    api_base: str,
    api_key: str,
    model: str,
    n_votes: int = 3,
) -> tuple[dict[str, int] | None, list[dict[str, int]]]:
    """Self-consistency: 跑 n 次 LLM, 逐类多数投票.

    Returns:
        (final_labels, all_votes) — final_labels 是多数投票结果,
        all_votes 是全部成功投票的列表.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 并行调用 n 次 (temperature=0.3 产生多样性)
    def _single_call() -> dict[str, int] | None:
        return call_llm(
            report, api_base, api_key, model,
            max_retries=1, temperature=0.3,
        )

    votes: list[dict[str, int]] = []
    with ThreadPoolExecutor(max_workers=n_votes) as executor:
        futures = [executor.submit(_single_call) for _ in range(n_votes)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                votes.append(result)

    if not votes:
        return None, []

    if len(votes) == 1:
        return votes[0], votes

    # 逐类多数投票 (平局时偏向 0, 即保守)
    final: dict[str, int] = {}
    for col in LABEL_COLS:
        count_1 = sum(v[col] for v in votes)
        final[col] = 1 if count_1 > len(votes) / 2 else 0

    return final, votes


def _build_messages(report: str) -> list[dict[str, str]]:
    """构建包含 few-shot 示例的消息列表."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for ex_report, ex_json in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex_report})
        messages.append({"role": "assistant", "content": ex_json})
    messages.append({"role": "user", "content": report})
    return messages


def call_llm(
    report: str,
    api_base: str,
    api_key: str,
    model: str,
    max_retries: int = 2,
    temperature: float = 0.0,
) -> dict[str, int] | None:
    """调用 LLM API, 返回解析后的 12 维标签, 失败返回 None."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": _build_messages(report),
        "temperature": temperature,
        "max_tokens": 4000,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code != 200:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [HTTP {resp.status_code}] {resp.text[:200]}")
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                content = ""
            content = content.strip()
            parsed = parse_llm_response(content)
            if parsed is not None:
                return parsed
            # JSON 解析失败, 打印原始返回方便排查
            if attempt >= max_retries:
                print(f"\n  [PARSE] raw={content[:300]!r}")
                if hasattr(resp, "text"):
                    print(f"  [PARSE] full_response={resp.text[:500]!r}")
            else:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  [ERROR] {e}")
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="NLP 伪标签 — 第一阶段验证")
    parser.add_argument(
        "--api-base",
        default=API_BASE,
        help="LLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=API_KEY,
        help="LLM API key",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="只跑前 N 个样本 (0=全部)",
    )
    parser.add_argument(
        "--train-csv",
        default="data/metadata/train.csv",
        help="竞赛标签文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="输出目录",
    )
    args = parser.parse_args()

    # ── 项目根目录 ──────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    train_csv = project_root / args.train_csv
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 58 个有标签的报告 ──────────────────────────────
    train = pd.read_csv(train_csv, encoding="utf-8")
    labeled_mask = train[LABEL_COLS].notna().all(axis=1)
    labeled_df = train[labeled_mask].copy()
    for col in LABEL_COLS:
        labeled_df[col] = labeled_df[col].astype(int)

    if args.limit > 0:
        labeled_df = labeled_df.head(args.limit)

    n_samples = len(labeled_df)
    print(f"加载 {n_samples} 个有标签报告")
    print(f"API: {args.api_base}  model: {args.model}")
    print(f"输出: {output_dir}")
    print()

    if not args.api_key or "YOUR_" in args.api_key:
        print("[ERROR] 请先在脚本顶部 API_KEY 处填写你的 DeepSeek API key")
        print("        获取: https://platform.deepseek.com/api_keys")
        sys.exit(1)

    # ── 逐样本调用 LLM ──────────────────────────────────────
    results = []
    n_ok = 0
    n_fail = 0

    for idx, (_, row) in enumerate(labeled_df.iterrows()):
        report = str(row["Report"])
        study_uid = row["StudyInstanceUID"]

        # 截断过长报告 (>3000 chars 通常已足够覆盖所有发现)
        report_truncated = report[:4000]

        print(f"[{idx+1}/{n_samples}] {study_uid[-12:]}... ", end="", flush=True)

        llm_labels = call_llm(
            report_truncated,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )

        if llm_labels is None:
            print("FAIL")
            n_fail += 1
            continue

        # 规则修正层: 用关键词纠正 LLM 明显错误 (0 API 成本)
        corrected_labels = apply_rule_correction(report, llm_labels)

        n_corrected = sum(
            1 for col in LABEL_COLS
            if corrected_labels[col] != llm_labels[col]
        )
        corr_info = f" corrected={n_corrected}" if n_corrected > 0 else ""

        print(f"OK{corr_info}")
        n_ok += 1

        true_labels = {col: int(row[col]) for col in LABEL_COLS}
        conf = score_confidence(report, corrected_labels)
        results.append(
            {
                "StudyInstanceUID": study_uid,
                "Report_snippet": report[:200],
                **{f"true_{col}": true_labels[col] for col in LABEL_COLS},
                **{f"pred_{col}": corrected_labels[col] for col in LABEL_COLS},
                **{f"conf_{col}": conf[col] for col in LABEL_COLS},
                "n_corrected": n_corrected,
            }
        )

    if not results:
        print("\n[FATAL] 没有成功解析的样本, 退出.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "pseudo_labels_valid.csv", index=False, encoding="utf-8")
    print(f"\n{n_ok}/{n_samples} 成功 ({n_fail} 失败)")
    total_corrected = sum(r.get("n_corrected", 0) for r in results)
    if total_corrected > 0:
        print(f"规则修正: 共翻转 {total_corrected} 个预测")
    print(f"逐样本对比已保存: {output_dir / 'pseudo_labels_valid.csv'}")

    # ── 计算 per-class 指标 ─────────────────────────────────
    y_true = np.array([[r[f"true_{col}"] for col in LABEL_COLS] for r in results])
    y_pred = np.array([[r[f"pred_{col}"] for col in LABEL_COLS] for r in results])

    print("\n" + "=" * 70)
    print("验证结果 — Per-Class Metrics")
    print("=" * 70)

    rows = []
    for i, col in enumerate(LABEL_COLS):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        n_pos = int(y_true[:, i].sum())
        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"
        print(
            f"  {col:<20s}  P={fmt(prec)}  R={fmt(rec)}  F1={fmt(f1)}  "
            f"Acc={fmt(acc)}  (n_pos={n_pos})"
        )
        rows.append(
            {
                "class": col,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
                "n_positive": n_pos,
            }
        )

    # ── 置信度分布 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("置信度分布 (所有预测)")
    print("=" * 70)
    conf_counts = {level: 0 for level in ["HIGH", "MEDIUM", "LOW", "REVIEW"]}
    for r in results:
        for col in LABEL_COLS:
            level = r.get(f"conf_{col}", "HIGH")
            conf_counts[level] = conf_counts.get(level, 0) + 1
    total_preds = len(results) * len(LABEL_COLS)
    for level in ["HIGH", "MEDIUM", "LOW", "REVIEW"]:
        n = conf_counts[level]
        print(f"  {level:<8s}: {n:4d} ({n/total_preds*100:5.1f}%)")

    # ── 置信度过滤后的指标 (只用 HIGH 置信度的预测) ──────────
    print("\n" + "=" * 70)
    print("置信度过滤后 — 仅 HIGH 置信度预测 vs 真值")
    print("=" * 70)
    rows_filtered = []
    for i, col in enumerate(LABEL_COLS):
        # 筛选该列置信度为 HIGH 或 REVIEW (LLM=0+HIGH) 的样本
        high_mask = np.array([
            r[f"conf_{col}"] in ("HIGH", "REVIEW")
            for r in results
        ])
        n_high = int(high_mask.sum())
        if n_high == 0:
            print(f"  {col:<20s}  (无 HIGH 置信度样本)")
            rows_filtered.append({
                "class": col, "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "accuracy": float("nan"),
                "n_high_conf": 0,
            })
            continue

        yt = y_true[high_mask, i]
        yp = y_pred[high_mask, i]
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        acc = accuracy_score(yt, yp)
        n_pos = int(yt.sum())
        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"

        # 同时统计被丢弃的 LOW/MEDIUM 里的错误
        low_med_mask = np.array([
            r[f"conf_{col}"] in ("LOW", "MEDIUM")
            for r in results
        ])
        n_discarded = int(low_med_mask.sum())
        n_discarded_errors = int((y_pred[low_med_mask, i] != y_true[low_med_mask, i]).sum()) if n_discarded > 0 else 0

        print(
            f"  {col:<20s}  P={fmt(prec)}  R={fmt(rec)}  F1={fmt(f1)}  "
            f"Acc={fmt(acc)}  (n={n_high}, n_pos={n_pos}, "
            f"discarded={n_discarded}, errors_in_discarded={n_discarded_errors})"
        )
        rows_filtered.append({
            "class": col,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "n_high_conf": n_high,
        })

    # ── 过滤后的总体指标 ──────────────────────────────────────
    high_or_review_mask = np.array([
        r[f"conf_{col}"] in ("HIGH", "REVIEW")
        for r in results for col in LABEL_COLS
    ])
    yt_flat_filtered = y_true.flatten()[high_or_review_mask]
    yp_flat_filtered = y_pred.flatten()[high_or_review_mask]
    n_kept = len(yt_flat_filtered)
    if n_kept > 0:
        filtered_acc = accuracy_score(yt_flat_filtered, yp_flat_filtered)
        # 计算被丢弃预测中的错误数
        discarded_mask = ~high_or_review_mask
        n_discarded = int(discarded_mask.sum())
        n_errors_discarded = int(
            (y_true.flatten()[discarded_mask] != y_pred.flatten()[discarded_mask]).sum()
        ) if n_discarded > 0 else 0
        print(f"\n  过滤后样本数: {n_kept}/{total_preds} ({n_kept/total_preds*100:.1f}%)")
        print(f"  过滤后 Accuracy: {filtered_acc:.3f}")
        print(f"  丢弃预测数: {n_discarded}  其中错误: {n_errors_discarded}")

    # ── REVIEW 统计 (LLM=0 但报告有高置信关键词 → 可能是假阴性) ──
    n_review_fp = 0
    n_review_fn = 0
    review_details = []
    for r in results:
        for col in LABEL_COLS:
            if r.get(f"conf_{col}") == "REVIEW":
                t = r[f"true_{col}"]
                p = r[f"pred_{col}"]
                if t == 1 and p == 0:
                    n_review_fn += 1
                    review_details.append(f"  [FN-REVIEW] {r['StudyInstanceUID'][-12:]} {col}: true=1 pred=0 (漏检!)")
                elif t == 0 and p == 0:
                    n_review_fp += 1  # LLM 正确判 0, 但我们标记了 REVIEW (过度标记)
    if review_details:
        print(f"\n  REVIEW 标记样本 (LLM=0 但含关键词): {conf_counts.get('REVIEW', 0)}")
        print(f"    其中真 FN (LLM 漏检): {n_review_fn}")
        print(f"    其中真 TN (过度标记): {n_review_fp}")
        for detail in review_details[:5]:
            print(detail)
        if len(review_details) > 5:
            print(f"    ... 还有 {len(review_details)-5} 条")

    # ── 总体 (未过滤) ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("总体指标 (未过滤, 含所有预测)")
    print("=" * 70)
    overall_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n  Overall accuracy: {overall_acc:.3f}")
    print(f"  Macro F1:         {macro_f1:.3f}")

    # ── 保存报告 ────────────────────────────────────────────
    metrics_df = pd.DataFrame(rows)
    report_path = output_dir / "nlp_validation_report.csv"
    metrics_df.to_csv(report_path, index=False)
    print(f"\nPer-class 指标已保存: {report_path}")

    filtered_path = output_dir / "nlp_validation_filtered.csv"
    pd.DataFrame(rows_filtered).to_csv(filtered_path, index=False)
    print(f"过滤后指标已保存: {filtered_path}")

    # ── 结论 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if overall_acc >= 0.85 and macro_f1 >= 0.75:
        print("[PASS] NLP 伪标签质量足够, 可以进入第二阶段批量标注.")
    elif overall_acc >= 0.75:
        print("[WARN] 边缘质量. 建议调优 prompt 后重试, 或用 HIGH/MEDIUM/LOW 分层.")
    else:
        print("[FAIL] 未通过. 需要分析失败样本, 重新设计 prompt 或更换模型.")
    print("=" * 70)


if __name__ == "__main__":
    main()
