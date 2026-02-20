from deepforest import main, visualize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

model = main.deepforest()

image_path = "/Users/terezasaskova/Downloads/bologna_3.png"

# Predict trees
print("Running prediction...")
predictions = model.predict_image(path=image_path)

if predictions is not None and len(predictions) > 0:
    print(f"\n Found {len(predictions)} trees!")
    print(f"Confidence range: {predictions['score'].min():.2f} - {predictions['score'].max():.2f}")
    
    # Save CSV
    predictions.to_csv("detected_trees.csv", index=False)
    print("Results saved to 'detected_trees.csv'")
    
    # Manual visualization
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(1, figsize=(20, 15))
    ax.imshow(img)
    
    # Draw bounding boxes
    for idx, row in predictions.iterrows():
        rect = Rectangle(
            (row['xmin'], row['ymin']),
            row['xmax'] - row['xmin'],
            row['ymax'] - row['ymin'],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add confidence score label
        ax.text(
            row['xmin'], row['ymin'] - 5,
            f"{row['score']:.2f}",
            color='red',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
    
    ax.axis('off')
    plt.title(f'Detected Trees in Bologna: {len(predictions)}', fontsize=16)
    plt.tight_layout()
    plt.savefig('bologna_trees_detected.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'bologna_trees_detected.png'")
    plt.show()
    
    # Print summary statistics
    print(f"\n Summary:")
    print(f"Total trees: {len(predictions)}")
    print(f"High confidence (>0.5): {len(predictions[predictions['score'] > 0.5])}")
    print(f"Medium confidence (0.3-0.5): {len(predictions[(predictions['score'] >= 0.3) & (predictions['score'] <= 0.5)])}")
    print(f"Low confidence (<0.3): {len(predictions[predictions['score'] < 0.3])}")
else:
    print("No trees detected")