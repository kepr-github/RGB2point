
from model import PointCloudNet
from utils import predict
import torch



model_save_name = "ckpt/mymodel.pth"

model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
model.load_state_dict(torch.load(model_save_name)["model"])
model.eval()  

image_path = "img/09.png"
save_path = "result/09_1.ply"

predict(model, image_path, save_path)
