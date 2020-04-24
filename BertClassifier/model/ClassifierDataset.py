from torch.utils.data.dataset import Dataset
import json



class BCDataset(Dataset):
    def __init__(self, train_file, model):
        with open(train_file) as f:
            items = f.readlines()
            items = [json.loads(item) for item in items]
        self.
