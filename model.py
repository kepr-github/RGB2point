import torch.nn as nn
from torch.nn import MultiheadAttention
import torch
import timm
class PointCloudGeneratorWithAttention(nn.Module):
    def __init__(
        self, input_feature_dim, point_cloud_size, num_heads=16, dim_feedforward=2048
    ):
        super(PointCloudGeneratorWithAttention, self).__init__()
        print(f"input_feature_dim:{input_feature_dim}")
        print(f"dim_feedforward:{dim_feedforward}")
        print(f"point_cloud_size:{point_cloud_size*3}")
        self.self_attention = MultiheadAttention(
            embed_dim=input_feature_dim, num_heads=num_heads
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(input_feature_dim, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(
                dim_feedforward, point_cloud_size * 3
            ),  # Output layer for point cloud
        )
        self.point_cloud_size = point_cloud_size

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_feature_dim]
        # Transpose for the attention layer
        x = x.transpose(0, 1)  # Shape: [seq_length, batch_size, input_feature_dim]

        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)

        # Transpose back
        attn_output = attn_output.transpose(
            0, 1
        )  # Shape: [batch_size, seq_length, input_feature_dim]

        # Pass through the linear layers
        point_cloud = self.linear_layers(attn_output.flatten(start_dim=1))

        # Reshape to (batch_size, point_cloud_size, 3)
        point_cloud = point_cloud.view(-1, self.point_cloud_size, 3)
        return point_cloud


class PointCloudNet(nn.Module):
    def __init__(self, num_views, point_cloud_size, num_heads, dim_feedforward, train_vit=False):
        super(PointCloudNet, self).__init__()
        self.train_vit = train_vit
        # Load the pretrained Vision Transformer model from timm
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        if not self.train_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        # Define the number of features from the ViT model
        num_features = self.vit.num_features

        # Aggregate features from different views
        out_features = 1024 * 4
        self.aggregator = nn.Linear(num_features, out_features)
        # Point cloud generator with attention
        self.point_cloud_generator = PointCloudGeneratorWithAttention(
            input_feature_dim=out_features,
            point_cloud_size=point_cloud_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, x):
        batch_size, num_views, C, H, W = x.shape

        # Process all views in the batch
        x = x.view(batch_size * num_views, C, H, W)

        # Extract features from the views using ViT
        if self.train_vit:
            features = self.vit(x)
        else:
            with torch.no_grad():
                features = self.vit(x)

        # Reshape features back to separate views
        features = features.view(batch_size, num_views, -1)

        # Compute the mean of features from all views
        mean_features = torch.mean(features, dim=1)

        # Aggregate features
        aggregated_features = self.aggregator(mean_features)
        aggregated_features = aggregated_features.unsqueeze(1)

        # Generate point cloud
        point_cloud = self.point_cloud_generator(aggregated_features)

        # Reshape to (batch_size, point_cloud_size, 3)
        point_cloud = point_cloud.view(batch_size, -1, 3)

        return point_cloud