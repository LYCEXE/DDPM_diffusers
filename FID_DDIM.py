from torchvision import transforms
from diffusers import DDPMPipeline,DDPMScheduler,DDIMScheduler
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
import os

# pipeline = DDPMPipeline.from_pretrained("celeba_train_2/train_2/97")
# pipeline.to("cuda")
# images = pipeline().images
# images[0].save(f"celebahq.png")
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
prefix = "DDIM"
if not os.path.exists(f"{root}/{prefix}"):
    os.mkdir(f"{root}/{prefix}")
unet = UNet2DModel.from_pretrained("cifar10_pipeline/google/unet")
unet.to("cuda")
schedule = DDIMScheduler()
schedule.set_timesteps(50)
pipeline = DDPMPipeline(unet, schedule)
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
    save_images(ground_truth, f"{prefix}/ground_truth", root, step)
    images = pipeline(batch_size = len(images[0])).images
    save_images_PIL(images, f"{prefix}/results", root, step)
# index = 0
# for image in images:
#     image.save(os.path.join(f"cifar10_fid/{index}.jpg"))
#     index+=1
# def make_grid(images, rows, cols):
#     w, h = images[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     for i, image in enumerate(images):
#         grid.paste(image, box=(i%cols*w, i//cols*h))
#     return grid
#
#
# def evaluate():
#     # Sample some images from random noise (this is the backward diffusion process).
#     # The default pipeline output type is `List[PIL.Image]`
#     pipeline = DDPMPipeline.from_pretrained("test_001").to("cuda")
#     images = pipeline(
#         batch_size = 16,
#         generator=torch.manual_seed(0),
#     ).images
#
#     # Make a grid out of the images
#     image_grid = make_grid(images, rows=4, cols=4)
#
#     # Save the images
#
#     image_grid.save(f"eval/{1:04d}.png")
#
# model = UNet2DModel.from_pretrained("test_001/unet")
# model.to("cuda")

# dataset = CIFAR10(
#     root="./dataset/cifar10",
#     train= False,
#     download= True,
#     transform= transforms.Compose(
#         [
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0],[1])
#         ]
#     )
# )

# dataset = ImageFolder(
#         root='./celeba_hq/train',
#         transform=transforms.Compose(
#             [
#                 transforms.Resize(256),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0,0,0], [1,1,1]),
#             ]
#         ))

# images = pipeline().images
# images[0].save("test1111.png")
# dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
# noise_scheduler = DDPMScheduler.from_config("test_001/scheduler")
# print(model.config)
# print(noise_scheduler)
# evaluate()

# for step,images in enumerate(dataloader):
#     ground_truth = images[0].to("cuda")
    # noise = torch.randn(images[0].shape).to("cuda")
    # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (32,), device="cuda").long()
    # noisy_images = noise_scheduler.add_noise(ground_truth, noise, timesteps)
    # pred_images = model(noisy_images, timesteps, return_dict=False)[0]
    # save_images(ground_truth,"ground_truth")
    # save_images(noisy_images,"noisy_images")
    # save_images(pred_images,"pred_images")
    # break




#FID:  33.33583877792091 DDIM_100