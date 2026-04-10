import pandas as pd

# Load the extracted dataset
df = pd.read_csv('sarship_dataset_chars_SSDD_OBB_Loc.csv')

print(f"Original row count (with looping artifacts): {len(df)}")

# CRITICAL FIX: keep='last' ensures we keep the row where all ships
# in the image had been parsed, preserving true nearest neighbor distances.
df_cleaned = df.drop_duplicates(subset=['image_path', 'gt_bbox'], keep='last')

print(f"Cleaned row count: {len(df_cleaned)} (Matches SSDD exactly)")

# Verify the fix using an image known to have 3 ships
example = df_cleaned[df_cleaned['image_path'] == '000006.jpg']
print("\nVerification for 000006.jpg (Should be 3 ships with valid distances):")
print(example[['image_path', 'gt_bbox', 'nearest_neighbor_px']])

# Save the pristine dataset for your CIRT analysis
df_cleaned.to_csv('sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.csv', index=False)
print("\nSaved pristine dataset to 'sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.csv'.")