""" https://github.com/vahidk/tfrecord

    Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s.
    Params:
    -------
    data_path: str
        The path to the tfrecords file.
    index_path: str or None
        The path to the index file.
    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.
    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.
"""
import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/physionet2012/1.0.10/physionet2012-test.tfrecord"
#Z:\daintlab\data\sr\paper\setfunction\tensorflow_datasets\root\tensorflow_datasets\physionet2012\1.0.10
index_path = None
description = None
dataset = TFRecordDataset(tfrecord_path, index_path, description)
print(type(dataset))
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

for i, data in enumerate(loader):
    print(len(loader))
    #x,y = data
#data = next(iter(loader))
#print(data)