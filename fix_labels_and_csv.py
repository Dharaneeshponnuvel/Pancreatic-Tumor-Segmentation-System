import os
import pandas as pd
import SimpleITK as sitk

# --------- Helper function to fix label dtype ---------
def fix_labels(input_csv, output_csv, fixed_label_dir):
    os.makedirs(fixed_label_dir, exist_ok=True)

    df = pd.read_csv(input_csv, header=None, names=["image", "label"])
    new_rows = []

    for _, row in df.iterrows():
        img_path = row["image"]
        lbl_path = row["label"]

        if not os.path.exists(lbl_path):
            print(f"❌ Missing label: {lbl_path}")
            continue

        # Load label
        lbl = sitk.ReadImage(lbl_path)
        arr = sitk.GetArrayFromImage(lbl)

        # Force to int16
        fixed_arr = arr.astype("int16")

        # Save new label
        base = os.path.basename(lbl_path)
        new_lbl_path = os.path.join(fixed_label_dir, base)
        fixed_lbl = sitk.GetImageFromArray(fixed_arr)
        fixed_lbl.CopyInformation(lbl)
        sitk.WriteImage(fixed_lbl, new_lbl_path)

        # Add row with new label path
        new_rows.append([img_path, new_lbl_path])

    # Save new CSV
    new_df = pd.DataFrame(new_rows, columns=["image", "label"])
    new_df.to_csv(output_csv, index=False, header=False)
    print(f"✅ Fixed CSV written: {output_csv}")


# --------- Paths ---------
BASE = "D:\\dharaneesh"
csv_dir = os.path.join(BASE, "csv_lists")

task1_train_csv = os.path.join(csv_dir, "task1_train.csv")
task1_val_csv   = os.path.join(csv_dir, "task1_val.csv")
task2_train_csv = os.path.join(csv_dir, "task2_train.csv")
task2_val_csv   = os.path.join(csv_dir, "task2_val.csv")

# Task1 + Task2 label dirs
task1_labels_fixed = os.path.join(BASE, "PANTHER_Task1", "LabelsTr_fixed")
task2_labels_fixed = os.path.join(BASE, "PANTHER_Task2", "LabelsTr_fixed")

# --------- Run fixing ---------
fix_labels(task1_train_csv, os.path.join(csv_dir, "task1_train_fixed.csv"), task1_labels_fixed)
fix_labels(task1_val_csv,   os.path.join(csv_dir, "task1_val_fixed.csv"),   task1_labels_fixed)
fix_labels(task2_train_csv, os.path.join(csv_dir, "task2_train_fixed.csv"), task2_labels_fixed)
fix_labels(task2_val_csv,   os.path.join(csv_dir, "task2_val_fixed.csv"),   task2_labels_fixed)

