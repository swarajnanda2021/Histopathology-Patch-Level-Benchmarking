import pandas as pd

df = pd.read_csv('/data1/vanderbc/vanderbc/MIL_CODE_GAMMA/publication_plots/data/reciprocal_encoder_metrics_summary.csv')

# Filter: UNI encoder, IMPACT training, cross-validation
uni_impact = df[(df['encoder'] == 'Uni') & 
                (df['training_dataset'] == 'IMPACT') & 
                (df['evaluation_type'] == 'cross_validation')]

# Select relevant columns
result = uni_impact[['tumor', 'gene', 'roc_auc_mean', 'roc_auc_std', 'splits']].copy()

# 95% CI = 1.96 * std (or 1.96 * std/sqrt(splits) if you want SEM-based CI)
result['ci_95'] = 1.96 * result['roc_auc_std']
result['formatted'] = result.apply(
    lambda r: f"{r['roc_auc_mean']:.3f} $\\pm$ {r['ci_95']:.3f}", axis=1
)

print(result[['tumor', 'gene', 'formatted']].to_string(index=False))
