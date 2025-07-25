from torch.utils.data import Dataset
from glob import glob
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class PCDataset(Dataset):
    def __init__(self, stage, transform=None, root="data", categories=None):
        """Dataset loader for point clouds and renderings.

        Parameters
        ----------
        stage: str
            Either ``"train"`` or ``"test"``.
        transform: torchvision transform or ``None``
            Applied to the loaded images.
        root: str, optional
            Root directory containing ``ShapeNet_pointclouds`` and
            ``ShapeNetRendering`` folders.
        categories: list of str, optional
            Category names to load. When ``None`` the default ShapeNet
            categories used by the original code are selected.
        """

        self.transform = transform
        self.stage = stage
        self.root = root

        default_categories = ["02958343", "02691156", "03001627"]
        self.categories = categories if categories is not None else default_categories

        # Optional train/test split text files. If they don't exist, build the
        # file list by scanning the dataset directories.
        if stage == "train":
            split_file = os.path.join(root, "shapenet_train.txt")
        elif stage == "test":
            split_file = os.path.join(root, "shapenet_test.txt")
        else:
            split_file = None

        self.filenames = []
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                self.filenames = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover available models
            for c in self.categories:
                cat_dir = os.path.join(root, "ShapeNet_pointclouds", c)
                if not os.path.isdir(cat_dir):
                    continue
                for label in os.listdir(cat_dir):
                    self.filenames.append(f"{c}/{label}")

        self.numbers_list = [f"{i:02}" for i in range(24)]

        labels = []
        category = set()
        for f in self.filenames:
            attr = f.split("/")
            labels.append(attr[1].strip())
            category.add(attr[0])

        category = list(category)
        self.labels = []
        self.data = []

        for c in self.categories:
            for label in labels:
                volume_path = os.path.join(
                    self.root,
                    "ShapeNet_pointclouds",
                    c,
                    label,
                    "pointcloud_2048.npy",
                )
                extensions = ["jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]
                files = []
                for ext in extensions:
                    files.extend(
                        glob(
                            os.path.join(
                                self.root,
                                "ShapeNetRendering",
                                c,
                                label,
                                "rendering",
                                f"*.{ext}",
                            )
                        )
                    )
                files.sort()
                for file in files:
                    if self.stage == "train":
                        if os.path.exists(volume_path):
                            self.data.append([c, label, file])

                # if self.stage == "test":
                #     if os.path.exists(volume_path) and len(files) > 1:
                #         test_image_path = os.path.join(
                #             self.root,
                #             "ShapeNetRendering",
                #             c,
                #             label,
                #             "rendering",
                #             "エンレイ3_1.JPG",
                #         )
                #         self.data.append([c, label, test_image_path])
                if self.stage == "test":
                    # 点群データとレンダリング画像が1枚以上存在するかチェック
                    if os.path.exists(volume_path) and len(files) > 0:
                        # filesリストの最初の画像パスをテスト用として使用
                        test_image_path = files[0]
                        self.data.append([c, label, test_image_path])
                        

    def __len__(self):
        return len(self.data)

    def normalize_point_cloud(self, point_cloud):
        """
        Normalize a point cloud to be centered around the origin and fit within a unit cube.

        :param point_cloud: Numpy array of shape (num_points, dimensions)
        :return: Normalized point cloud.
        """
        centroid = np.mean(point_cloud, axis=0)
        centered_point_cloud = point_cloud - centroid
        if self.stage == "train":
            np.random.shuffle(centered_point_cloud)
        return centered_point_cloud

    def __getitem__(self, idx):
        data = self.data[idx]
        category = data[0]
        label = data[1]
        image = data[2]

        image_files = [image]
        pc = np.load(
            os.path.join(
                self.root,
                "ShapeNet_pointclouds",
                category,
                label,
                "pointcloud_2048.npy",
            )
        )
        pc = self.normalize_point_cloud(pc)

        images = []
        for filename in image_files:
            image = Image.open(filename).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        name = f"{category}_{label}"
        images_tensor = torch.stack(images, dim=0)

        return images_tensor, torch.as_tensor(pc, dtype=torch.float32), name
    


def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist

def fscore(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
    
class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, pred, target):
        # pred and target are expected to have shape (batch_size, 1024, 3)
        assert pred.shape == target.shape
        assert pred.shape[1] == 1024 and pred.shape[2] == 3

        batch_size = pred.shape[0]
        num_points = pred.shape[1]

        # Compute pairwise distances between all points
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        dist = torch.sum(diff**2, dim=-1)

        # Solve the assignment problem using Hungarian algorithm
        # Note: This is a simplified version and may not be the most efficient for large point clouds
        assignment = torch.zeros_like(dist)
        for b in range(batch_size):
            _, indices = torch.topk(dist[b], k=num_points, largest=False, dim=1)
            assignment[b] = torch.scatter(assignment[b], 1, indices, 1)

        # Compute the EMD
        emd = torch.sum(dist * assignment, dim=[1, 2]) / num_points

        return emd.mean()


def export_to_ply(point_cloud, filename):
    """Save a point cloud to ``filename`` in ASCII PLY format.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Array of shape ``(num_points, 3)`` containing xyz coordinates.
    filename : str
        Output file path.
    """

    num_points = point_cloud.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header\n",
    ]

    with open(filename, "w") as f:
        f.write("\n".join(header))
        np.savetxt(f, point_cloud, fmt="%f %f %f")


from torchvision import transforms
from PIL import Image
def predict(model, image_path, save_path):

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    input_tensor = transform(image)
    input_tensor = input_tensor.reshape(1,1,3,224,224)

    

    # Invoke the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)


    export_to_ply(output[0], save_path)
    print(f"Image from {image_path} saved to {save_path}")