import yaml
import torch
from models.unet_cond_base import Unet  # Adjust the import path if needed
dir_name = "/umbc/rs/zzbatmos/users/ztushar1/StableDiffusion-PyTorch-bision/"
# Load the config file
with open(dir_name+"config/image_cond.yaml", "r") as f:
    config = yaml.safe_load(f)

model_config = config["ldm_params"]

# Dummy input setup
batch_size = 2
im_channels = config["dataset_params"]["im_channels"]
image_size = config["dataset_params"]["im_size"]
timesteps = config["diffusion_params"]["num_timesteps"]
device = "cpu"

# Initialize model
model = Unet(im_channels=im_channels, model_config=model_config).to(device)

# Create dummy inputs
x = torch.randn(batch_size, im_channels, image_size, image_size).to(device)
t = torch.randint(0, timesteps, (batch_size,), dtype=torch.long).to(device)

# Dummy conditioning input for image/text condition
cond_input = {}
if model.image_cond:
    cond_input["image"] = torch.randn(batch_size, 
                                    model_config["condition_config"]["image_condition_config"]["image_condition_input_channels"],
                                    model_config["condition_config"]["image_condition_config"]["image_condition_h"],
                                    model_config["condition_config"]["image_condition_config"]["image_condition_w"]).to(device)
# if model.text_cond:
#     cond_input["text"] = torch.randn(batch_size, 
#                                     model_config["condition_config"]["text_condition_config"]["text_embed_dim"]).to(device)

# Run the model
with torch.no_grad():
    out = model(x, t, cond_input if model.cond else None)
    print(f"Output shape: {out.shape}")  # Should be (batch_size, im_channels, image_size, image_size)

# # print(model)
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of parameters: {:,}".format(num_params))  