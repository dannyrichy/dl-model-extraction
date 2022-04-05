import urllib

import torch
from torchvision import transforms


class Model:
    model = None

    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def sample(self):
        from PIL import Image
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        self.output(input_batch)

    def output(self, input_batch):
        with torch.no_grad():
            output = self.model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)
