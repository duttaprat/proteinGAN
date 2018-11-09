#!/usr/bin/env python3

import argparse
import h5py

DATASET_NAME = "Analyses/Basecall_1D_000/BaseCalled_template/Events"

parser = argparse.ArgumentParser(description='Removing rows from events dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, help='path of FAST5 file')
parser.add_argument('n_lines_to_delete', type=int, help='Number of lines to delete from beginning')

args = parser.parse_args()

print("Opening {} file".format(args.path))
hdf5_file = h5py.File(args.path, 'r+')
print("Deleting first {} rows. Original dataset contains {} rows".format(args.n_lines_to_delete,
                                                                         hdf5_file.get(DATASET_NAME).shape[0]))

updated_events = hdf5_file.get(DATASET_NAME)[args.n_lines_to_delete:]

del hdf5_file['Analyses/Basecall_1D_000/BaseCalled_template/Events']
hdf5_file.create_dataset('Analyses/Basecall_1D_000/BaseCalled_template/Events', data=updated_events)

print("After deleting dataset contains {} rows".format(hdf5_file.get(DATASET_NAME).shape[0]))
hdf5_file.close()
