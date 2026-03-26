# How to Run IC-Knock-Poly Advantage Experiments (CORRECTED)

## ✅ All Code is Now Corrected

All experiments now use **CORRECT polynomial term evaluation** [base, exponent] instead of just base feature indices.

## 🚀 Quick Start

### 1. Rebuild Rust (if needed)
```bash
cd /Users/haoyi/Desktop/CIKnockoffPolyReg/rust
cargo build --release
```

### 2. Run Quick Advantage Demo (2-3 minutes)
```bash
cd /Users/haoyi/Desktop/CIKnockoffPolyReg
python simulation_results/quick_advantage_demo.py
```

This will show:
- Test 1: n=50, p=10, k=2 (Moderate dimension)
- Test 2: n=50, p=15, k=2 (Higher dimension)  
- Test 3: Semi-supervised n=30+500, p=10, k=2

**Output format:**
```
Ground truth: [[0, 2], [1, 1]]  # x_0², x_1¹
IC-Knock-Poly:
  True terms:     [[0, 2], [1, 1]]
  Selected:       [[0, 2], [1, 2]]  # x_0², x_1²
  Correct:        1/2
  FDR: 0.500 | TPR: 0.500
```

### 3. Run Full Comparison (5-10 minutes)
```bash
cd /Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results
python run_quick_simulation.py
```

### 4. Run Complete Simulation Suite (30-60 minutes)
```bash
cd /Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results
python run_simulations.py
```

## 🎯 What Will You See?

### IC-Knock-Poly's 4 Key Advantages:

1. **ITERATIVE SELECTION** (vs One-Shot)
   - Multiple passes recover features missed initially
   - Better performance in high dimensions (p >> n)

2. **POSI ALPHA-SPENDING** (vs Fixed FDR)
   - Adaptive threshold: q_t = Q × 6/(π²t²)
   - Provable FDR control across iterations
   - Better power than conservative knockoff+

3. **GMM MODELING** (vs Single Gaussian)
   - Handles multi-modal distributions
   - Better covariance estimation
   - More accurate knockoffs

4. **SEMI-SUPERVISED** (vs Supervised only)
   - Uses unlabeled data for better distribution estimation
   - Critical when n is small
   - Improves power without inflating FDR

## 📊 Example Output

```
======================================================================
Test 1: n=50, p=10, k=2 (Moderate dimension)
======================================================================
Ground truth: [[0, 2], [1, 1]]

IC-Knock-Poly:
  True terms:     [[0, 2], [1, 1]]
  Selected:       [[0, 2], [1, 1]]
  Correct:        2/2
  FDR: 0.000 | TPR: 1.000

Poly-Knockoff:
  True terms:     [[0, 2], [1, 1]]
  Selected:       [[0, 2], [1, 2]]
  Correct:        1/2
  FDR: 0.500 | TPR: 0.500

Poly-CLIME:
  True terms:     [[0, 2], [1, 1]]
  Selected:       [[0, 2], [1, 1]]
  Correct:        2/2
  FDR: 0.000 | TPR: 1.000
```

## 🔍 Key Insight

**Before (WRONG):**
- Only compared base feature indices: {0, 1}
- Selected x_0² and x_1³ → "2/2 correct" ❌

**After (CORRECT):**
- Compare exact polynomial terms: [base, exponent]
- Selected x_0² and x_1³ → "1/2 correct" ✓
- Wrong exponent matters!

## 📁 Result Files

After running, results are saved to:
- `simulation_results/default_sweep_summary.json`
- `simulation_results/default_sweep_summary.csv`
- `simulation_results/figures/*.pdf`

## 🎓 Citation

When using these experiments, cite:
- Correct polynomial term evaluation [base, exponent]
- PoSI alpha-spending sequence
- GMM-based conditional knockoffs

---

**Ready to run!** Start with `quick_advantage_demo.py` for a quick test.
