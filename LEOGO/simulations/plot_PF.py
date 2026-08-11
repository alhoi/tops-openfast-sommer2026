import pandas as pd
import matplotlib.pyplot as plt
import os


leogo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
folder_path = os.path.join(leogo_path, 'PF_files')
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]



for file_path in file_paths:

    df_raw = pd.read_csv(file_path, header=None)
    labels = df_raw.iloc[0, 1:].tolist()
    units = df_raw.iloc[1, :].tolist()
    data = df_raw.iloc[2:].reset_index(drop=True)
    data = data.apply(pd.to_numeric)
    time = data.iloc[:, 0]

    plt.figure()
    for i, name in enumerate(labels):
        plt.plot(time, data.iloc[:, i+1], label=name)
    plt.xlabel(f'{units[0]}')
    plt.ylabel(f'{units[1]}') #assuming all data after time will be of the same type
    plt.legend(labels)
    plt.title(str.split(file_path, '/')[-1].replace('.csv', ''))
    plt.grid()
    plt.show()


