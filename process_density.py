import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Read the parquet file
df = pd.read_parquet('density.parquet')

# Create the scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(df['fibro'], df['pred'], alpha=0.5)
plt.xlabel('Percent Fibroglandular Tissue', fontsize=16)
plt.ylabel('Density Score', fontsize=16)
plt.title('Synthetic Data Predictions', fontsize=20)

# Set equal aspect ratio for x and y axes
plt.axis('equal')

# Get the current axis limits
ax = plt.gca()
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]

# Plot the dashed diagonal line y=x
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

# Update layout for better readability
plt.tick_params(axis='both', which='major', labelsize=14)

# Save as SVG and PNG
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / 'synthetic_data_predictions.svg', bbox_inches='tight')
plt.savefig(output_dir / 'synthetic_data_predictions.png', dpi=300, bbox_inches='tight')

plt.close()

print("Plots saved in the 'output' directory.")
