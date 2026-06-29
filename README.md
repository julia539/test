## Step 3: Interim Scoring (ML Ensemble)

### Rationale

Each ML model captures different aspects of protein fitness:
- **ProteinMPNN**: Structural compatibility and stability
- **ESM-2**: Evolutionary conservation and naturalness
- **SaProt**: Combined structural and evolutionary context

The ensemble approach combines these complementary signals for more robust predictions.

### Model Contributions

**1. ProteinMPNN** (0.60 weight)

**What it evaluates:** Structural compatibility and stability under thermal stress

**Why it gets the highest weight:** ProteinMPNN directly assesses whether a sequence can fold into the target structure. It is trained on thousands of protein structures and learns the relationship between sequence and structure. The model's sensitivity to backbone perturbations (via noise analysis) makes it particularly valuable for predicting thermostability. Mutations that maintain high scores across multiple noise levels are likely to be stable at elevated temperatures.

**2. SaProt** (0.25 weight)

**What it evaluates:** Combined structural and evolutionary context

**Why it gets the second highest weight**: SaProt uniquely bridges the gap between structure-based and sequence-based methods. By encoding structural features directly into the input, it captures both evolutionary patterns (like ESM-2) and structural constraints (like ProteinMPNN). This hybrid approach provides complementary information that neither purely structural nor purely sequence-based models can capture alone.

**3. ESM-2** (0.15 weight)

**What it evaluates:** Evolutionary conservation and sequence naturalness

**Why it gets the lowest weight:** While ESM-2 excels at identifying sequences that match evolutionary patterns, it does not directly consider protein structure. A sequence can be evolutionarily "natural" but still contain mutations that destabilize the structure. Thus, ESM-2 provides useful but secondary information compared to structure-aware models.

### Method

**1. Normalize scores**

All scores are normalized using percentile rank:

```python
def percentile_rank(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Percentile rank normalization. Set reverse=True when lower is better."""
    rank = series.rank(pct=True)
    return 1.0 - rank if reverse else rank
```

**Why percentile rank?**
- Scores from different models are on different scales
- Percentile rank transforms scores to [0,1] range
- Better for combining scores than raw values

**2. Compute individual scores**

**MPNN score** combines base and stability scores:

```python
df["mpnn_base_norm"] = percentile_rank(df["base_score"], reverse=True)
df["mpnn_stability_norm"] = percentile_rank(df["stability_score"], reverse=True)

df["MPNN_score"] = (
    MPNN_INTERNAL["base"] * df["mpnn_base_norm"] +
    MPNN_INTERNAL["stability"] * df["mpnn_stability_norm"]
)
```

| Component | Weight | Rationale |
|-----------|--------|-----------|
| base_score | 0.6 | Structural compatibility at baseline |
| stability_score | 0.4 | Stability under thermal stress |

**SaProt score** is the normalized saprot_score:

```python
df["SaProt_score"] = percentile_rank(df["saprot_score"], reverse=False)
```

**ESM score** is the normalized esm_score (reversed because lower is better):

```python
df["ESM_score"] = percentile_rank(df["esm_score"], reverse=True)
```

**3. Ensemble weights**

```python
WEIGHTS = {
    "MPNN": 0.60,
    "SaProt": 0.25,
    "ESM": 0.15,
}
```

| Model | Weight | Rationale |
|-------|--------|-----------|
| ProteinMPNN | 0.60 | Direct structural evaluation |
| SaProt | 0.25 | Combined structural-language |
| ESM-2 | 0.15 | Only sequence-based |

**Why these weights?**
- ProteinMPNN provides the most direct evaluation of structural stability
- SaProt adds valuable structural context beyond sequence
- ESM-2 is useful but less predictive than structure-based methods
- The weights reflect each model's expected contribution to predicting thermostability

**4. Ensemble score calculation**

```
Ensemble_score = 0.60 * MPNN_score + 0.25 * SaProt_score + 0.15 * ESM_score
```

**5. Candidate selection**

Candidates are sorted by Ensemble_score, and the top 50 are selected for further analysis.

### Script

```python
import logging
import pandas as pd
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MPNN_FILE = PROJECT_DIR / "step2.1-proteinmpnn" / "results-proteinmpnn.csv"
ESM_FILE = PROJECT_DIR / "step2.2-esm" / "results-esm.csv"
SAPROT_FILE = PROJECT_DIR / "step2.3-saprot" / "results-saprot.csv"

OUTPUT_FILE = Path("top-50-candidates.csv")
TOP_N = 50

WEIGHTS = {
    "MPNN": 0.60,
    "SaProt": 0.25,
    "ESM": 0.15,
}

MPNN_INTERNAL = {"base": 0.6, "stability": 0.4}


def load_all_scores() -> pd.DataFrame:
    """Load MPNN, SaProt and ESM scores, merge on Sequence."""
    mpnn = pd.read_csv(MPNN_FILE)
    saprot = pd.read_csv(SAPROT_FILE)
    esm = pd.read_csv(ESM_FILE)

    if "sequence" in mpnn.columns:
        mpnn = mpnn.rename(columns={"sequence": "Sequence"})

    merged = (
        mpnn
        .merge(saprot[["Sequence", "saprot_score"]], on="Sequence", how="inner")
        .merge(esm[["Sequence", "esm_score"]], on="Sequence", how="inner")
    )
    logger.info("Merged %d candidates", len(merged))
    return merged


def percentile_rank(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Percentile rank normalization. Set reverse=True when lower is better."""
    rank = series.rank(pct=True)
    return 1.0 - rank if reverse else rank


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized score columns for each model."""
    df["mpnn_base_norm"] = percentile_rank(df["base_score"], reverse=True)
    df["mpnn_stability_norm"] = percentile_rank(df["stability_score"], reverse=True)

    df["MPNN_score"] = (
        MPNN_INTERNAL["base"] * df["mpnn_base_norm"] +
        MPNN_INTERNAL["stability"] * df["mpnn_stability_norm"]
    )
    df["SaProt_score"] = percentile_rank(df["saprot_score"], reverse=False)
    df["ESM_score"] = percentile_rank(df["esm_score"], reverse=True)
    return df


def ensemble_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted ensemble score and rank candidates."""
    df["Ensemble_score"] = (
        WEIGHTS["MPNN"] * df["MPNN_score"] +
        WEIGHTS["SaProt"] * df["SaProt_score"] +
        WEIGHTS["ESM"] * df["ESM_score"]
    )
    return df.sort_values("Ensemble_score", ascending=False)


def select_top(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Keep top N candidates and assign sequential IDs."""
    top = df.head(n).copy()
    top.insert(0, "candidate_id", range(1, n + 1))
    return top


def main():
    df = load_all_scores()
    df = normalize_scores(df)
    df = ensemble_score(df)
    top50 = select_top(df)

    top50.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved %d candidates to %s", len(top50), OUTPUT_FILE)

    report_cols = [
        "candidate_id", "Group", "Mutations",
        "MPNN_score", "SaProt_score", "ESM_score", "Ensemble_score"
    ]
    present = [c for c in report_cols if c in top50.columns]
    print(top50[present].to_string(index=False))


if __name__ == "__main__":
    main()
```

### Output Files

**top-50-candidates.csv** contains:
- All columns from 500-candidates.csv
- Normalized scores for each model
- Ensemble_score
- Candidate selection criteria

## Step 4: FoldX Analysis

### Tool Overview

**FoldX** (Schymkowitz et al., 2005) is an empirical force field for protein stability analysis.

**Key features:**
- Fast calculation (seconds per mutation)
- Accurate ΔΔG predictions
- Comprehensive energy decomposition
- Widely used in protein engineering

**Energy terms in FoldX:**
1. Van der Waals interactions
2. Electrostatics
3. Hydrogen bonds (backbone and sidechain)
4. Solvation (polar and hydrophobic)
5. Entropy (sidechain and backbone)
6. Clashes (van der Waals)
7. Torsional energy
8. Water bridges
9. Disulfide bonds
10. Ionization energy

### Methodology

**1. Prepare mutation list**

The script `create-indlst.py` converts mutations from trimmed indexing to FoldX-compatible format:

```python
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
csv_path = BASE_DIR / 'step3-interim-scoring' / 'top-50-candidates.csv'

with open(csv_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    new_mutations_list = []
    for row in reader:
        new_mutations = []
        mutations = row['Mutations'].split(';')
        for mut in mutations:
            number = int(''.join([n for n in mut if n.isdigit()]))
            if number < 64:
                number +=1
            if number > 64:
                number+=3
            new_mut = mut[0] + 'A' + str(number) + mut[-1]
            new_mutations.append(new_mut)
        new_mutations_str = ','.join(new_mutations) + ';' + '\n'
        new_mutations_list.append(new_mutations_str)

with open('individual_list.txt', 'w') as file:
    file.writelines(new_mutations_list)
```

**Why this conversion?**
- FoldX uses full-length PDB numbering
- Our trimmed sequence omits M1 and replaces TYG with X
- Positions need to be shifted to match PDB

**2. Run FoldX BuildModel**

The script `run.sh` executes FoldX:

```bash
#!/bin/bash
./foldx_20270131 -c BuildModel --pdb 2B3P_new.pdb --mutant-file individual_list.txt --numberOfRuns 5 | tee mutations.log
```

**Parameters:**
- `BuildModel`: Mode for modeling mutations
- `--pdb`: Input structure file
- `--mutant-file`: File containing mutation lists
- `--numberOfRuns 5`: Number of independent runs for stochastic averaging

**Why 5 runs?**
- FoldX uses stochastic algorithms
- Multiple runs provide more reliable estimates
- 5 runs balance accuracy and computational cost

**3. Parse and score results**

The script `add-foldx-results.py` processes FoldX output:

```python
import logging
import re
import pandas as pd
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "step3-interim-scoring" / "top-50-candidates.csv"
FXOUT_PATH = SCRIPT_DIR.parent / "step4-foldx" / "Dif_2B3P_new.fxout"
OUTPUT_PATH = Path("top-10-final-score.csv")

FOLDX_WEIGHTS = {
    "total_energy": 0.55,
    "clashes": 0.20,
    "entropy": 0.15,
    "electrostatics": 0.05,
    "solvation": 0.05,
}
ENSEMBLE_WEIGHT = 0.7
FOLDX_WEIGHT = 0.3

FOLDX_COLUMNS = [
    "Pdb", "total_energy", "Backbone_Hbond", "Sidechain_Hbond",
    "Van_der_Waals", "Electrostatics", "Solvation_Polar",
    "Solvation_Hydrophobic", "Van_der_Waals_clashes",
    "entropy_sidechain", "entropy_mainchain", "sloop_entropy",
    "mloop_entropy", "cis_bond", "torsional_clash",
    "backbone_clash", "helix_dipole", "water_bridge", "disulfide",
    "electrostatic_kon", "partial_covalent_bonds",
    "energy_Ionisation", "Entropy_Complex",
]


def load_top50(path: Path) -> pd.DataFrame:
    """Load top-50 CSV, ensure candidate_id and ensemble column exist."""
    df = pd.read_csv(path)
    logger.info("CSV candidates: %d", len(df))

    if "candidate_id" not in df.columns:
        raise ValueError("Missing 'candidate_id' column")

    ensemble_col = next(
        (c for c in ["Ensemble_score", "ensemble_score", "final_score"] if c in df.columns),
        None
    )
    if ensemble_col is None:
        raise ValueError(f"No ensemble column found. Available: {list(df.columns)}")
    logger.info("Using ensemble column: '%s'", ensemble_col)
    return df, ensemble_col


def parse_fxout(path: Path) -> pd.DataFrame:
    """Parse FoldX .fxout file into a DataFrame."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("2B3P"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                values = [float(x) for x in parts[1:]]
            except (ValueError, IndexError):
                continue
            rows.append([parts[0]] + values)

    fx = pd.DataFrame(rows, columns=FOLDX_COLUMNS[: len(rows[0])])
    logger.info("FoldX structures: %d", len(fx))
    return fx


def extract_candidate_id(pdb_name: str) -> tuple[int, int]:
    """Parse candidate_id and replica from '2B3P_new_{id}_{replica}.pdb'."""
    match = re.search(r"_(\d+)_(\d+)\.pdb", pdb_name)
    if not match:
        raise ValueError(f"Cannot parse FoldX name: {pdb_name}")
    return int(match.group(1)), int(match.group(2))


def average_replicates(fx: pd.DataFrame) -> pd.DataFrame:
    """Add candidate_id/replica, then average numeric columns per candidate."""
    parsed = fx["Pdb"].apply(extract_candidate_id)
    fx["candidate_id"] = parsed.apply(lambda x: x[0])
    fx["replica"] = parsed.apply(lambda x: x[1])

    replica_counts = fx.groupby("candidate_id").size()
    if not (replica_counts == 5).all():
        bad = replica_counts[replica_counts != 5].to_dict()
        logger.warning("Some candidates lack 5 replicas: %s", bad)

    logger.info("Unique candidates in FoldX: %d", fx["candidate_id"].nunique())

    numeric_cols = [c for c in fx.columns if c not in ("Pdb", "candidate_id", "replica")]
    return fx.groupby("candidate_id")[numeric_cols].mean().reset_index()


def percentile_norm(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Percentile rank normalization. reverse=True when lower values are better."""
    rank = series.rank(pct=True)
    return 1.0 - rank if reverse else rank


def add_foldx_score(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FoldX metrics and compute weighted foldx_score."""
    metrics = {
        "foldx_total": ("total_energy", True),
        "foldx_clash": ("Van_der_Waals_clashes", True),
        "foldx_entropy": ("entropy_sidechain", True),
        "foldx_electro": ("Electrostatics", True),
        "foldx_solv": ("Solvation_Hydrophobic", True),
    }
    for new_col, (src_col, reverse) in metrics.items():
        df[new_col] = percentile_norm(df[src_col], reverse=reverse)

    df["foldx_score"] = (
        FOLDX_WEIGHTS["total_energy"] * df["foldx_total"] +
        FOLDX_WEIGHTS["clashes"] * df["foldx_clash"] +
        FOLDX_WEIGHTS["entropy"] * df["foldx_entropy"] +
        FOLDX_WEIGHTS["electrostatics"] * df["foldx_electro"] +
        FOLDX_WEIGHTS["solvation"] * df["foldx_solv"]
    )
    return df


def main():
    df, ensemble_col = load_top50(CSV_PATH)
    fx = parse_fxout(FXOUT_PATH)
    fx_avg = average_replicates(fx)

    missing = set(df["candidate_id"]) - set(fx_avg["candidate_id"])
    if missing:
        raise ValueError(f"Missing FoldX candidates: {missing}")
    extra = set(fx_avg["candidate_id"]) - set(df["candidate_id"])
    if extra:
        logger.warning("Extra FoldX candidates not in CSV: %s", extra)

    df = df.merge(fx_avg, on="candidate_id", how="inner")
    logger.info("After merge: %d", len(df))

    df = add_foldx_score(df)

    df["final_score"] = (
        ENSEMBLE_WEIGHT * df[ensemble_col] + FOLDX_WEIGHT * df["foldx_score"]
    )

    corr_ens_foldx = df[ensemble_col].corr(df["foldx_score"])
    corr_ens_total = df[ensemble_col].corr(df["total_energy"])
    logger.info("Corr(%s, foldx_score)=%.3f", ensemble_col, corr_ens_foldx)
    logger.info("Corr(%s, total_energy)=%.3f", ensemble_col, corr_ens_total)

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "final_rank", range(1, len(df) + 1))

    top10 = df.head(10)
    top10.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved top-10 to %s", OUTPUT_PATH)

    report_cols = [
        "final_rank", "candidate_id", "Group", "Num_mutations", "Mutations",
        ensemble_col, "foldx_score", "final_score", "total_energy",
    ]
    present = [c for c in report_cols if c in df.columns]
    print(top10[present].to_string(index=False))

    logger.info("Total candidates ranked: %d", len(df))


if __name__ == "__main__":
    main()
```

**FoldX components and weights:**

| Component | Weight | Description |
|-----------|--------|-------------|
| Total Energy | 0.55 | Overall stability (lower is better) |
| Van der Waals clashes | 0.20 | Steric strain (lower is better) |
| Sidechain entropy | 0.15 | Conformational cost (lower is better) |
| Electrostatics | 0.05 | Charge interactions (lower is better) |
| Solvation | 0.05 | Hydrophobic interactions (lower is better) |

**Total Energy** (0.55)

Total energy is the most important indicator of protein stability. It represents the overall energetic cost of folding the protein into its native structure. Lower total energy corresponds to a more stable protein. This component captures the cumulative effect of all interactions and is the primary metric used in protein stability engineering.

**Van der Waals Clashes** (0.20)

Van der Waals clashes occur when atoms are too close together, creating steric strain. These are highly destabilizing because they represent unfavorable repulsive interactions. Minimizing clashes is critical for protein stability, as even a single bad clash can significantly destabilize a protein. This component is particularly important when introducing multiple mutations.

**Sidechain Entropy** (0.15)

Sidechain entropy represents the conformational freedom of amino acid sidechains. When a protein folds, sidechains lose entropy, and this must be compensated by favorable interactions. Mutations that reduce sidechain entropy (e.g., replacing a flexible residue with a rigid one) can be stabilizing by reducing the entropic cost of folding. However, excessive rigidity can also be destabilizing, making this a nuanced component.

**Electrostatics** (0.05)

Electrostatic interactions include charge-charge and charge-dipole interactions. While important for protein stability, these interactions are often context-dependent and can be either stabilizing or destabilizing. The relatively low weight reflects that:

Electrostatic calculations can be less accurate than other energy terms

Electrostatic effects are often secondary to hydrophobic packing and hydrogen bonding

Many proteins can tolerate electrostatic changes without large stability effects

**Solvation** (0.05)

Solvation energy represents the cost of transferring hydrophobic residues from water into the protein interior. While hydrophobic packing is critical for stability, solvation effects are often captured indirectly by other energy terms. The low weight reflects that solvation is generally well-optimized in stable proteins and mutations rarely cause large solvation changes.

### Output Files

**Dif_2B3P_new.fxout** contains:
- PDB name
- Energy components for each run
- Average and standard deviation

**mutations.log** contains:
- FoldX execution details
- Mutation processing information
- Errors and warnings

**individual_list.txt** contains:
- Mutation lists in FoldX format
- One line per candidate
- Format: oldAAnumberAnewAA
