""" A script to convert a tsv file to a poses file.
"""

import argparse
import pandas as pd
import os
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a tsv file to a poses file.')
    parser.add_argument('tsv_file', type=str, help='The tsv file to convert.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tsv_file = args.tsv_file
    df = pd.read_csv(tsv_file, sep='\t')
    print(df)
    """
         Unnamed: 0         x         y         z        qx        qy        qz        qw
    0             0  0.442987  0.057997 -0.299888  0.010416 -0.017658 -0.008972  0.999750
    1             1  0.445180  0.058046 -0.297291  0.010236 -0.017754 -0.008999  0.999749
    2             2  0.448238  0.057964 -0.297758  0.010310 -0.017719 -0.008871  0.999750
    """

    # Normalize
    mean_x = df['x'].mean()
    mean_y = df['y'].mean()
    mean_z = df['z'].mean()
    df['x'] -= mean_x
    df['y'] -= mean_y
    df['z'] -= mean_z
    norm = (df['x']**2 + df['y']**2 + df['z']**2)**0.5
    norm_max = norm.max()
    df['x'] /= norm_max
    df['y'] /= norm_max
    df['z'] /= norm_max
    print(df)

    # Convert the tsv file to a poses file.
    save_dir = f"{os.path.dirname(tsv_file)}/pose"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(df)):
        x = df['x'][i]
        y = df['y'][i]
        z = df['z'][i]
        qx = df['qx'][i]
        qy = df['qy'][i]
        qz = df['qz'][i]
        qw = df['qw'][i]
        r = Rotation.from_quat([qx, qy, qz, qw])
        mat = r.as_matrix()
        pose_file = f"{save_dir}/{i:06d}.txt"
        with open(pose_file, 'w') as f:
            f.write(f"{mat[0][0]} {mat[0][1]} {mat[0][2]} {x}\n")
            f.write(f"{mat[1][0]} {mat[1][1]} {mat[1][2]} {y}\n")
            f.write(f"{mat[2][0]} {mat[2][1]} {mat[2][2]} {z}\n")
            f.write("0 0 0 1\n")
