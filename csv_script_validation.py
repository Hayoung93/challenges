import sys
import pandas as pd

IMAGE_COL_CANDIDATES = ('image_name', 'image', 'file', 'filename')
SCORE_COL_CANDIDATES = ('score', 'prob', 'probability', 'label')


def check_submission_csv(path):
    print(f"Checking file: {path}")

    # 1. Can the CSV be read at all?
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("CSV cannot be read.")
        print("Error:", e)
        return

    print("CSV loaded successfully")

    # 2. Check image column
    image_col = next((c for c in IMAGE_COL_CANDIDATES if c in df.columns), None)
    if image_col is None:
        print("Missing image column.")
        print(f"Expected one of: {IMAGE_COL_CANDIDATES}")
        print("Found columns:", list(df.columns))
        return
    print(f"Image column found: '{image_col}'")

    # 3. Check score column
    score_col = next((c for c in SCORE_COL_CANDIDATES if c in df.columns), None)
    if score_col is None:
        print("Missing score column.")
        print(f"Expected one of: {SCORE_COL_CANDIDATES}")
        print("Found columns:", list(df.columns))
        return
    print(f"Score column found: '{score_col}'")

    # 4. Check for missing values
    if df[image_col].isnull().any():
        print("Image column contains missing values")
        return

    if df[score_col].isnull().any():
        print("Score column contains missing values")
        return

    # 5. Check score column is numeric
    try:
        df[score_col].astype(float)
    except ValueError:
        print("Score column cannot be converted to float")
        print("Sample values:", df[score_col].head().tolist())
        return

    # 6. Check duplicate image names
    if df[image_col].duplicated().any():
        print("Duplicate image names found")
        duplicates = df[df[image_col].duplicated()][image_col].unique()
        print("Duplicates:", duplicates[:10])
        return

    print("CSV format is VALID")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_submission.py <submission.csv>")
    else:
        check_submission_csv(sys.argv[1])
