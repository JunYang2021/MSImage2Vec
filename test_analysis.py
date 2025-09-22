import pandas as pd

r"""# Processing of ad study data (8 clusters)
# embedding mzs
mz_groups = pd.read_csv(r'E:\yangjun\msi\MSI_IIE_article\ad_study\kmeans_clustering_8\mz_groups_clusters.csv')
pos_cols_only = [col for col in mz_groups.columns if 'pos' in col.lower()]
pos_columns = ['mean_mz'] + pos_cols_only
mz_groups_pos = mz_groups[pos_columns].copy()
mz_groups_pos = mz_groups_pos.dropna(subset=pos_cols_only, how='all')

neg_cols_only = [col for col in mz_groups.columns if 'neg' in col.lower()]
neg_columns = ['mean_mz'] + neg_cols_only
mz_groups_neg = mz_groups[neg_columns].copy()
mz_groups_neg = mz_groups_neg.dropna(subset=neg_cols_only, how='all')

# identified compounds
identified_pos = pd.read_excel(r'E:\yangjun\msi\ad_msi\Qualitative.xlsx', sheet_name='pos')
pos_mz = identified_pos['mz'].astype(float).tolist()
pos_name = identified_pos['Metabolites'].astype(str).tolist()
identified_neg = pd.read_excel(r'E:\yangjun\msi\ad_msi\Qualitative.xlsx', sheet_name='neg')
neg_mz = identified_neg['mz'].astype(float).tolist()
neg_name = identified_neg['Metabolites'].astype(str).tolist()


def annotate_mz(df, mz_list, name_list):
    annotations = []
    for mz in df['mean_mz']:
        matched_names = []
        for target_mz, target_name in zip(mz_list, name_list):
            ppm_tol = target_mz * 10e-6  # 10 ppm tolerance
            if abs(mz - target_mz) <= ppm_tol:
                matched_names.append(target_name)
        annotations.append('; '.join(matched_names) if matched_names else '')
    df['compound_name'] = annotations
    return df

# Annotate mz groups
mz_groups_pos = annotate_mz(mz_groups_pos, pos_mz, pos_name)
mz_groups_neg = annotate_mz(mz_groups_neg, neg_mz, neg_name)

# Save annotated data to Excel
output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_study\kmeans_clustering_8\annotated_mz_groups.xlsx'
with pd.ExcelWriter(output_path) as writer:
    mz_groups_pos.to_excel(writer, sheet_name='positive', index=False)
    mz_groups_neg.to_excel(writer, sheet_name='negative', index=False)"""


# Processing of ad study data (4 clusters)
# embedding mzs
mz_groups = pd.read_csv(r'E:\yangjun\msi\MSI_IIE_article\ad_study\kmeans_clustering_4\mz_groups_clusters.csv')
pos_cols_only = [col for col in mz_groups.columns if 'pos' in col.lower()]
pos_columns = ['mean_mz'] + pos_cols_only
mz_groups_pos = mz_groups[pos_columns].copy()
mz_groups_pos = mz_groups_pos.dropna(subset=pos_cols_only, how='all')

neg_cols_only = [col for col in mz_groups.columns if 'neg' in col.lower()]
neg_columns = ['mean_mz'] + neg_cols_only
mz_groups_neg = mz_groups[neg_columns].copy()
mz_groups_neg = mz_groups_neg.dropna(subset=neg_cols_only, how='all')

# identified compounds
identified_pos = pd.read_excel(r'E:\yangjun\msi\ad_msi\Qualitative.xlsx', sheet_name='pos')
pos_mz = identified_pos['mz'].astype(float).tolist()
pos_name = identified_pos['Metabolites'].astype(str).tolist()
identified_neg = pd.read_excel(r'E:\yangjun\msi\ad_msi\Qualitative.xlsx', sheet_name='neg')
neg_mz = identified_neg['mz'].astype(float).tolist()
neg_name = identified_neg['Metabolites'].astype(str).tolist()


def annotate_mz(df, mz_list, name_list):
    annotations = []
    for mz in df['mean_mz']:
        matched_names = []
        for target_mz, target_name in zip(mz_list, name_list):
            ppm_tol = target_mz * 10e-6  # 10 ppm tolerance
            if abs(mz - target_mz) <= ppm_tol:
                matched_names.append(target_name)
        annotations.append('; '.join(matched_names) if matched_names else '')
    df['compound_name'] = annotations
    return df

# Annotate mz groups
mz_groups_pos = annotate_mz(mz_groups_pos, pos_mz, pos_name)
mz_groups_neg = annotate_mz(mz_groups_neg, neg_mz, neg_name)

# Save annotated data to Excel
output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_study\kmeans_clustering_4\annotated_mz_groups.xlsx'
with pd.ExcelWriter(output_path) as writer:
    mz_groups_pos.to_excel(writer, sheet_name='positive', index=False)
    mz_groups_neg.to_excel(writer, sheet_name='negative', index=False)
