import pandas as pd
import os

"""
Processes a video journaling dataset and saves the processed data as a CSV file.

The script filters and processes the dataset based on specified columns, sorts the data,
and attempts to match video files based on `webscreen_id` and `Trigger.Index`. The resulting
dataset includes the file paths to the matched videos.

Usage: datasetup.py --input_csv <input_csv> --output_csv <output_csv> --video_folder <video_folder> [--webscreen_ids <webscreen_id1> <webscreen_id2> ...]

Arguments:
    --input_csv       Path to the input CSV file containing the dataset.
    --output_csv      Path to the output CSV file where the processed data will be saved.
    --video_folder    Folder path containing the videos.
    --webscreen_ids   Optional list of webscreen_ids to filter, leave empty to include all.

Note: The --webscreen_ids argument is optional. If not provided, the script will process all webscreen_ids.
"""

def process_data(df, webscreen_ids=None):
    filtered_columns = ['ema_aware', 'ema_support', 'ema_insight', 'ema_fulfilled', 'ema_hopeless', 'ema_anxious', 'webscreen_id', 'Trigger.Index']
    df_filtered = df[filtered_columns]
    df_filtered = df_filtered.dropna()
    df_filtered.sort_values(by=['Trigger.Index', 'webscreen_id'], ascending=[True, True], inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    
    if webscreen_ids:
        df_filtered = df_filtered[df_filtered['webscreen_id'].isin(webscreen_ids)]
    
    return df_filtered

def get_file_path(row, folder_path):
    webscreen_id = int(row['webscreen_id'])
    trigger_index = str(int(row['Trigger.Index'])).zfill(2)

    mov_file = f"{webscreen_id}_{trigger_index}.MOV"
    mp4_file = f"{webscreen_id}_{trigger_index}.mp4"

    sub_dir = os.path.join(folder_path, str(webscreen_id))

    if not os.path.exists(sub_dir):
        return None

    for root, dirs, files in os.walk(sub_dir):
        if mov_file in files:
            return os.path.join(root, mov_file)
        elif mp4_file in files:
            return os.path.join(root, mp4_file)

    return None

def main(input_csv="../data/emogo_full_study.csv", output_csv="../data/file_paths.csv", video_folder="../data/video", webscreen_ids=None):
    df = pd.read_csv(input_csv, low_memory=False)
    print("read csv")
    df_final = process_data(df, webscreen_ids)
    print("processed")
    df_final['file_path'] = df_final.apply(get_file_path, axis=1, folder_path=video_folder)
    print("added file paths")
    df_final.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    main()