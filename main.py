import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = [] # list of features
        
        for layer_num, layer in enumerate(self.model):
            x = layer(x) # forward pass
            if str(layer_num) in self.chosen_features: # if layer is in chosen features
                features.append(x)
        
        return features
    

def load_image(file_path):
    image = Image.open(file_path) # PIL image
    image = loader(image).unsqueeze(0) # add the batch dimension
    return image.to(device) # use GPU if available

def save_image(tensor, file_name):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image) # convert tensor to PIL image
    image.save("images/generated/" + file_name)

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)), # resize image
    transforms.ToTensor() # convert image to Tensor
])
unloader = transforms.ToPILImage() # convert tensor to PIL image

# load images
content_img = load_image("images/content/cat.jpg")
style_img = load_image("images/style/monet2.jpg")

# generated = torch.randn(content_img.shape, device=device, requires_grad=True) # randomly initialize the image
generated = content_img.clone().requires_grad_(True) + 0.1*torch.randn(content_img.shape, device=device, requires_grad=True) # randomly initialize the image

save_image(generated, "generated.png")