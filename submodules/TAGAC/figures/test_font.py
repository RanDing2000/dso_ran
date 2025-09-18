import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Test current font setup
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})

# Create a simple test plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Font Test')

# Print current font information
current_font = plt.rcParams['font.family']
print(f"Current font family: {current_font}")

# List available serif fonts
serif_fonts = [f.name for f in fm.fontManager.ttflist if 'serif' in f.name.lower()]
print(f"Available serif fonts: {sorted(set(serif_fonts))}")

# Save the test plot
plt.savefig('font_test.png', bbox_inches='tight', dpi=300)
print("Test plot saved as font_test.png")
plt.close()

