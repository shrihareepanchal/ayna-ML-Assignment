import os, json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

color_vocab = {
    "red": 0, "green": 1, "blue": 2, "yellow": 3,
    "orange": 4, "purple": 5, "magenta": 6, "cyan": 7
}

class PolygonDataset(Dataset):
    def __init__(self, data_json_path, input_dir, output_dir):
        with open(data_json_path) as f:
            self.data = json.load(f)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        input_path = os.path.join(self.input_dir, entry['input_polygon'])
        output_path = os.path.join(self.output_dir, entry['output_image'])
        color_name = entry['colour'].lower()

        input_img = self.transform(Image.open(input_path).convert("RGB"))
        output_img = self.transform(Image.open(output_path).convert("RGB"))
        
        color_idx = color_vocab[color_name]
        return input_img, output_img, color_idx
