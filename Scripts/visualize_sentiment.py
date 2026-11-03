import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic styling for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# Load processed dataset
input_path = os.path.join('Data', 'processed', 'customer_tickets_processed.csv')
df = pd.read_csv(input_path)

print("Generating Sentiment & Customer Support Analytics Visualizations...")

# Create figures output folder
output_dir = os.path.join('dashboards')
os.makedirs(output_dir, exist_ok=True)

# 1. Sentiment Distribution by Category Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Category', hue='Sentiment_Label', palette='viridis')
plt.title('Customer Support Ticket Volume & Sentiment by Category', fontsize=14, fontweight='bold')
plt.xlabel('Ticket Category', fontweight='bold')
plt.ylabel('Ticket Count', fontweight='bold')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sentiment_by_category.png'), dpi=300)
plt.close()

# 2. Priority vs Sentiment Distribution Heatmap
plt.figure(figsize=(8, 5))
ct = pd.crosstab(df['Priority'], df['Sentiment_Label'], normalize='index') * 100
sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", cbar=True)
plt.title('Ticket Priority vs Sentiment Distribution (%)', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment Label', fontweight='bold')
plt.ylabel('Priority Level', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'priority_sentiment_heatmap.png'), dpi=300)
plt.close()

# 3. Word Count / Token Length Boxplot across Categories
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Category', y='Word_Count', palette='crest')
plt.title('Ticket Word Count Feature Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontweight='bold')
plt.ylabel('Word Count per Ticket', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'word_count_distribution.png'), dpi=300)
plt.close()

print(f"Visualizations successfully generated and saved into '{output_dir}/' folder!")