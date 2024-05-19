from torch.utils.data import Dataset

class VolvoDataset(Dataset):
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.groups = []
        self.streams = []

        with open(self.data_path, 'r') as fdata:
            for row in fdata:
                row = row.strip('\n').split('\t')
                self.uids.append(int(row[0]))
                self.streams.append(list(map(int, row[1:])))

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid, stream = self.uids[idx], self.streams[idx]
        return uid, stream