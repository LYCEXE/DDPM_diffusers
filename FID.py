from torchvision import transforms
from diffusers import DDPMPipeline
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import os

def save_images_PIL(images, prefix, root, step):
    for index, image in enumerate(images):
        if not os.path.exists(f"{root}/{prefix}"):
            os.mkdir(f"{root}/{prefix}")
        image.save(f"{root}/{prefix}/{step}_{index:04d}.png")

def save_images(images, prefix, root, step):
    for index, image in enumerate(images):
        unloader = transforms.ToPILImage()
        image = unloader(image)
        if not os.path.exists(f"{root}/{prefix}"):
            os.mkdir(f"{root}/{prefix}")
        image.save(f"{root}/{prefix}/{step}_{index:04d}.png")

root = "FID_dir"
pipeline = DDPMPipeline.from_pretrained("cifar10_pipeline/google")
pipeline.to("cuda")
dataset = CIFAR10(root='./dataset/cifar10',
                  train= False,
                  download= True,
                  transform= transforms.Compose(
        [
            transforms.ToTensor(),
        ])
                  )
dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
for step,images in enumerate(dataloader):
    ground_truth = images[0]
    save_images(ground_truth, "ground_truth", root, step)
    images = pipeline(batch_size = len(images[0])).images
    save_images_PIL(images, "results", root, step)


