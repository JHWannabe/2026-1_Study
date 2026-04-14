import pandas as pd

# Load files from Excel
file_path = r"C:\Users\user\Desktop\Study\data\AEC\강남\강남_merged_features.xlsx"
metadata = pd.read_excel(file_path, sheet_name="metadata")
features = pd.read_excel(file_path, sheet_name="features")



# Inspect structures
print("Metadata Info:")
print(metadata.info())
print(metadata.head())

print("\nFeatures Info:")
print(features.info())
print(features.head())

# Merge metadata and features
df = pd.merge(metadata[['PatientID', 'PatientAge', 'PatientSex', 'TAMA']], features, on='PatientID')

# Drop columns that are IDs or not features
feature_cols = [col for col in df.columns if col not in ['PatientID', 'TAMA', 'PatientAge', 'PatientSex']]

# Calculate correlation with TAMA
correlations = df[feature_cols + ['TAMA']].corr()['TAMA'].sort_values(ascending=False)

# Remove TAMA itself from correlations
correlations = correlations.drop('TAMA')

# Display top 20 and bottom 10 correlations
print("Top 20 correlations with TAMA:")
print(correlations.head(20))
print("\nBottom 10 correlations with TAMA:")
print(correlations.tail(10))

# Since TAMA is highly dependent on sex, let's look at correlations per sex
corr_male = df[df['PatientSex'] == 'M'][feature_cols + ['TAMA']].corr()['TAMA'].drop('TAMA').sort_values(ascending=False)
corr_female = df[df['PatientSex'] == 'F'][feature_cols + ['TAMA']].corr()['TAMA'].drop('TAMA').sort_values(ascending=False)

print("\nTop 10 features for Males:")
print(corr_male.head(10))

print("\nTop 10 features for Females:")
print(corr_female.head(10))

# Save the merged data and correlations for the user
df.to_csv('merged_data_for_analysis.csv', index=False)
correlations.to_csv('feature_tama_correlations.csv')


import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Top features to plot
top_features = ['p25', 'signal_energy', 'CV', 'min']

plt.figure(figsize=(15, 10))

for i, feature in enumerate(top_features):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(data=df, x=feature, y='TAMA', hue='PatientSex', alpha=0.5)
    plt.title(f'TAMA vs {feature}')

plt.tight_layout()
plt.savefig('top_features_vs_tama.png')

# Correlate features among themselves to check redundancy
redundancy_check = df[['p25', 'p10', 'signal_energy', 'AUC', 'median', 'min', 'mean']].corr()
print("\nCorrelation between top features (Redundancy Check):")
print(redundancy_check)