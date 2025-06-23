
from model import PointCloudNet
from utils import predict
import torch



model_save_name = "ckpt/pc1024_three.pth"

model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
model.load_state_dict(torch.load(model_save_name)["model"])
model.eval()  

image_path = "img/1013.jpg"
save_path = "result/1013.ply"

predict(model, image_path, save_path)
