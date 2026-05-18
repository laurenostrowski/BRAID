# BRAID:
This repository provides BRAID (**Behaviorally relevant Analysis of Intrinsic Dynamics**) from the Shanechi lab, with bugs fixed and patches to source code

Parsa Vahidi, Omid G. Sani, and Maryam Shanechi. *BRAID: Input-driven nonlinear dynamical modeling of neural-behavioral data.* ***In The Thirteenth International Conference on Learning
Representations***, 2025. URL https://openreview.net/forum?id=3usdM1AuI3.

## Changes to source code:
1. CRITICAL: MainModelPrepareArgs parser in BRAID/MainModel.py: **silent linear model is called when you think you called nonlinear MLP:**

regex character class: pythonregex = r"([A|K|Cy|Cz|A1|K1|Cy1|Cz1|A2|K2|Cy2|Cz2|]*)(\d+)HL(\d+)U" intended as alternation (A|K|Cy|Cz|...) but wrote it inside [ ], which makes it a character class. So the first group matches single characters from the set {A, |, K, C, y, z, 1, 2}, not the multi-character names

for-loop / if-statement indentation: hidden_layers, hidden_units outside the loop, so only the last regex match's var_names survives the loop – the if-block sets at most one component's _args to NL and the rest are set to {} (meaining linear)

**Fix: changed source code to parse model args correctly**

3. Stage 3 args inheritance (same issue with linear/non-linear models not be initiated correctly: **was making A3, K3, Cz3 linear always:**

Didn’t change source code, fixed by always passing stage 3 argos explicitly when model is fit: A3_args=NL_args, K3_args=NL_args, Cz3_args=NL_args

5. Patch to source code to enable noUZ feedthrough control:

I needed this in order to have a model where input can impact spiking activity directly (u → y feedthrough) but not behavior directly (NO u → z feedthrough). With this patch, when noUZ us set to True, input impacts behavior through the direct pathway (u → x1 → z) as well as an indirect pathway with a hidden layer (u → x3 → z)

*use noUZ=True when fitting model to enable*
