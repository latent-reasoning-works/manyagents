# Drug Discovery Notebooks for ManyAgents

**Branch**: `upper_bound`  
**Purpose**: Educational notebooks demonstrating agentic AI for drug discovery

---

## 📚 The 5 Learning Assets

### 1. **01_decoder_for_biologists.ipynb** ✅ COMPLETE
**What**: Interactive dictionary translating ML jargon → biological concepts  
**For**: Biologists with no AI background  
**Key Features**:
- ML-to-biology term translations
- Live examples with drug data
- Interactive term explorer
- Biological analogies for every concept

**Run**: Start here to understand the terminology!

### 2. **02_drug_target_discovery.ipynb** 
**What**: Real success story - how AI accelerates drug repurposing  
**For**: Seeing agents in action  
**Demonstrates**:
- Multi-agent workflow (data → scoring → validation)
- Using manyagents utils
- Real drug discovery pipeline

### 3. **03_agent_failure_modes.ipynb**
**What**: What happens when agents go wrong  
**For**: Understanding limitations and validation  
**Shows**:
- Overfitting failure mode
- Target bias failure mode  
- How validation agents catch problems
- Why experimental validation matters

### 4. **04_first_multiagent_analysis.ipynb**
**What**: Hands-on tutorial for running your first analysis  
**For**: Getting started with manyagents  
**Covers**:
- Traditional vs agentic approaches
- Step-by-step workflow setup
- Interpreting results
- Common pitfalls

### 5. **05_manual_vs_agentic.ipynb**
**What**: Side-by-side comparison of workflows  
**For**: Understanding the value proposition  
**Compares**:
- Time: manual (hours) vs agentic (minutes)
- Code: 50+ lines vs 5 lines of config
- Reproducibility: scattered scripts vs documented config
- Scalability: one-off vs reusable

---

## 🏗️ Architecture: How Notebooks Use ManyAgents

### Clean Separation

```
manyagents/ (on main branch)
├── adapters/              # Model wrappers (manylatents, cellforge, etc.)
├── utils/                 # Generic operations (data, scoring, validation)
└── examples/
    └── drug_discovery/    # Domain-specific functions
        ├── data.py        # fetch_chembl(), fetch_expression()
        ├── scoring.py     # score_repurposing()
        └── validation.py  # validate_safety()

notebooks/ (on upper_bound branch)
├── 01_decoder_for_biologists.ipynb      # Educational
├── 02_drug_target_discovery.ipynb       # Demo
├── 03_agent_failure_modes.ipynb         # Failure analysis
├── 04_first_multiagent_analysis.ipynb   # Hands-on
└── 05_manual_vs_agentic.ipynb           # Comparison
```

### How Notebooks Use the Infrastructure

**Pattern 1: Direct util usage**
```python
from manyagents.utils import data_ops, scoring, validation
from manyagents.examples.drug_discovery import data as drug_data

# Load data
chembl_data = drug_data.fetch_chembl(query="kinase_inhibitors", limit=500)

# Score candidates
scored = await scoring.score_and_rank(
    data=chembl_data,
    scoring_function="manyagents.examples.drug_discovery.scoring:score_repurposing",
    threshold=0.6
)
```

**Pattern 2: Through example functions**
```python
from manyagents.examples.drug_discovery import scoring, validation

# Use pre-built scoring
hypotheses = scoring.score_repurposing(
    data=merged_data,
    method="combined",
    failure_mode=None  # or "overfit" for demo
)

# Validate
safe = validation.validate_safety(
    data=hypotheses,
    checks=["toxicity", "ddi"]
)
```

---

## 🚀 Getting Started

### Quick Start
```bash
# Switch to upper_bound branch
git checkout upper_bound

# Start with notebook 1
jupyter notebook notebooks/01_decoder_for_biologists.ipynb
```

### Full Sequence
1. **01_decoder** - Learn the language
2. **02_drug_target** - See it in action  
3. **03_failure_modes** - Understand limitations
4. **04_first_analysis** - Try it yourself
5. **05_manual_vs_agentic** - Understand the value

**Total time**: ~2 hours to complete all 5

---

## 🔄 Branch Strategy

### Why Separate Branches?

**main branch**:
- Core infrastructure (adapters, utils)
- Stable, production-ready code
- Shared across all use cases

**upper_bound branch**:
- Educational notebooks
- Experimental demonstrations
- Domain-specific examples
- Can be messy/exploratory

### Syncing Changes

**Pull utils from main into upper_bound**:
```bash
git checkout upper_bound
git merge main --no-commit
# Review changes, keep only utils/adapters updates
git commit -m "Sync utils from main"
```

**Never merge notebooks back to main** - they stay on upper_bound!

---

## 📊 What Each Notebook Teaches

| Notebook | Concept | Biological Analogy | Key Takeaway |
|----------|---------|-------------------|--------------|
| 01 | Terminology | Lab equipment | AI terms = biology concepts |
| 02 | Success case | Drug repurposing | Agents accelerate discovery |
| 03 | Failure modes | False positives | Always validate! |
| 04 | Hands-on | Running assays | You can do this! |
| 05 | Comparison | Manual vs automated | 10x faster, more reproducible |

---

## 💡 For Instructors

### Teaching Tips

1. **Start with 01** - Don't skip the terminology!
2. **Show 03 early** - Failure modes build trust
3. **Let them run 04** - Hands-on is critical
4. **Use 05 for discussions** - Trade-offs, limitations

### Common Questions

**Q**: "Will AI replace drug discovery scientists?"  
**A**: No! Notebooks show AI as assistant, not replacement. You still need expertise for experimental design, validation, interpretation.

**Q**: "How do I know if results are real?"  
**A**: Notebook 03 covers this - always validate experimentally, check literature, use multiple methods.

**Q**: "Can I use this for my own data?"  
**A**: Yes! Notebook 04 shows how. Replace drug_data functions with your own loaders.

---

## 🎯 Learning Outcomes

After completing these notebooks, you will:

✅ Understand agentic AI terminology in biological context  
✅ Know how to use manyagents utils for drug discovery  
✅ Recognize failure modes and validation strategies  
✅ Be able to run your first multi-agent analysis  
✅ Understand when to use agentic vs traditional approaches  

---

## 📝 Notes for Developers

### Adding New Notebooks

1. Create on `upper_bound` branch
2. Use existing utils (don't duplicate)
3. Reference drug_discovery examples
4. Include failure modes (builds trust!)
5. Add to this README

### Modifying Utils

1. Make changes on `main` branch
2. Test with existing notebooks
3. Merge to `upper_bound`:
   ```bash
   git checkout upper_bound
   git merge main
   ```

### Data Sources

All data in these notebooks is **MOCK** data with realistic structure:
- `fetch_chembl()` - Mimics ChEMBL API responses
- `fetch_expression()` - Mimics LINCS L1000 format
- `fetch_drugbank()` - Mimics DrugBank structure

For production use, replace with real API calls (see function docstrings).

---

## 🤝 Contributing

Want to add a notebook?

1. Fork and create feature branch from `upper_bound`
2. Follow existing patterns (utils usage, clear explanations)
3. Include biological analogies
4. Add failure mode examples
5. Update this README
6. Submit PR to `upper_bound` branch

---

**Questions?** Open an issue or contact the manyagents team!
