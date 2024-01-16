# %%
import wfdb
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import spectrogram

directory_path = "/Users/princepatel/mit/classes/6.S898/6s898/scratchpaper/physionet.org/files/hd-semg/1.0.0/pr_dataset/"
sampling_rate = 2048
num_channels = 256
saved_data = pd.DataFrame(
    columns=["record_name", "subject_session", "raw", "preprocess"]
)


# def process_sigs(entry):
#     if isinstance(entry, list) and len(entry) == 1 and isinstance(entry[0], np.ndarray):
#         return entry[0]
#     else:

#         return entry

# %% Process each pair of .dat and .hea files
count = 0
for folder in os.scandir(directory_path):
    count += 1
    if folder.is_dir():
        folder_path = os.path.join(directory_path, folder.name) + "/"
        file_list = [f for f in os.listdir(folder_path) if f.endswith(".hea")]

        for file_name in file_list:
            record_name = os.path.splitext(file_name)[0]
            sample_name = record_name.split("_")

            if sample_name[0] == "maintenance":
                continue

            sample_type = sample_name[-2]
            record, _ = wfdb.rdsamp(folder_path + record_name)
            data = record.T

            record_name = sample_name[0] + "_" + sample_name[2]
            matching_records = saved_data.query(
                "record_name == @record_name & subject_session == @folder.name"
            )

            if not matching_records.empty:
                saved_data.loc[matching_records.index, sample_type] = [data]
            else:
                new_row = {
                    "record_name": record_name,
                    "subject_session": folder.name,
                    sample_type: data,
                }
                new_row = {col: new_row.get(col, None) for col in saved_data.columns}
                saved_data = pd.concat(
                    [saved_data, pd.DataFrame([new_row])], ignore_index=True
                )
    if count == 2:
        break

# %%
saved_data.to_pickle(
    "/Users/princepatel/mit/classes/6.S898/6s898/scratchpaper/saved_data_short.pkl"
)
# %%
temp = pd.read_pickle(
    "/Users/princepatel/mit/classes/6.S898/6s898/scratchpaper/saved_data_short.pkl"
)
# %%
rows = []
for _, row in temp.iterrows():
    raw_samples = row["raw"]
    preprocess_samples = row["preprocess"]

    for i in range(len(raw_samples)):
        raw_samp = np.array(raw_samples[i])
        preprocess_samp = np.array(preprocess_samples[i])
        _, _, raw_samp = spectrogram(raw_samp, fs=sampling_rate, nperseg=64, noverlap=28)
        _, _, preprocess_samp = spectrogram(preprocess_samp, fs=sampling_rate, nperseg=64, noverlap=28)
        print(raw_samp.shape)
        raw_samp = (raw_samp[:14,:])
        preprocess_samp = (preprocess_samp[:14,:])

        # Append rows to the list
        rows.append({"raw": raw_samp, "preprocess": preprocess_samp})

short_data = pd.DataFrame(rows)

all_arrays = np.concatenate([np.concatenate(short_data['raw'].values),
                             np.concatenate(short_data['preprocess'].values)])

global_mean = all_arrays.mean()
global_std = all_arrays.std()

short_data['raw'] = short_data['raw'].apply(lambda x: (x - global_mean) / global_std)
short_data['preprocess'] = short_data['preprocess'].apply(lambda x: (x - global_mean) / global_std)
# %%
short_data.to_pickle(
    "/Users/princepatel/mit/classes/6.S898/6s898/scratchpaper/short_data.pkl"
)
# %%
