# Test Pipeline Run — 2026-01-17

## Summary
- **City**: Meridian Reach (from test corpus facts)
- **Run scope**: Corpus (existing) → Facts (existing) → Dataset (existing) → Fine-tune → Evaluate
- **Base model**: Qwen/Qwen3-0.6B (from BASE_MODEL in .env)
- **Training mode**: QLoRA
- **OS**: Windows
- **Python**: 3.13.11 (venv)

## City snapshot (Meridian Reach)
Selected facts extracted from the test corpus:
- **Location**: City at the mouth of the River Sable and the Broadwater.
- **Harbor**: A bowl carved into stone with breakwaters angled toward the open sea.
- **Orchard Slopes**: Terraced landscape overlooking water and windward hills.
- **Civic core**: Market Square as the heart; Granaries as the lungs of the city.
- **Flood management**: Network of channels and culverts guiding floodwater toward the estuary.

## Dataset generated
From dataset_test/dataset_metadata.json:
- **QA pairs**: 50
- **Instruction pairs**: 25
- **Total examples**: 75
- **Train**: 67 examples
- **Test**: 8 examples

## Evaluation results
From eval/results/evaluation_results.json:
- **ROUGE-1**: 0.2381
- **ROUGE-2**: 0.0838
- **ROUGE-L**: 0.1599

### Example 1
**Prompt** (test example):
Question: Describe the climate and microclimate features of Meridian Reach.

**Reference**:
Meridian Reach has a temperate climate. Fog forms at dawn, especially near the harbor edge where a salt wind blows from the sea. The vegetation includes evergreen conifers, and citrus trees have a late fruiting season.

**Model prediction** (excerpt):
Meridian Reach is a coastal region ... characterized by its temperate maritime climate ... coastal fog that influences the weather patterns.

### Example 2
**Prompt** (test example):
Question: What topographic feature characterizes Orchard Slopes?

**Reference**:
Orchard Slopes features a terraced topography—a stepped landscape with terraces that overlook both water and windward hills, providing elevated views toward the water and the hills facing the prevailing winds.

**Model prediction** (excerpt):
Orchard Slopes is characterized by the presence of orchard ridges and steep slopes...

## Notes
- The run completed successfully and produced model artifacts in finetune/models/final and evaluation outputs in eval/results/.
- Warnings observed were non-fatal (tokenizer padding alignment, deprecation notices).
