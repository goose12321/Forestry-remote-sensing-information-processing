import open3d as o3d
import numpy as np
import laspy
from sklearn.cluster import DBSCAN

# 加载 .las 点云文件
def load_point_cloud(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

file_path = "CP30_ScanPos001-SINGLESCANS-220322_115025.las"
pcd = load_point_cloud(file_path)

# 可视化点云
o3d.visualization.draw_geometries([pcd], window_name="Loaded Point Cloud")

# 点云预处理，去噪与降采样
def preprocess_point_cloud(pcd, voxel_size=0.05):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # 降采样
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # 去噪
    return pcd

pcd = preprocess_point_cloud(pcd)
o3d.visualization.draw_geometries([pcd], window_name="Preprocessed Point Cloud")

# 使用DBSCAN进行单株分割
def segment_trees(pcd, eps=0.5, min_points=30):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_

    tree_point_clouds = []
    for i in range(labels.max() + 1):
        mask = labels == i
        tree_points = points[mask]
        tree_pcd = o3d.geometry.PointCloud()
        tree_pcd.points = o3d.utility.Vector3dVector(tree_points)
        tree_point_clouds.append(tree_pcd)
    return tree_point_clouds

trees = segment_trees(pcd)

# 可视化单株分割结果
for i, tree in enumerate(trees):
    tree.paint_uniform_color(np.random.rand(3))  # 给每棵树分配随机颜色
o3d.visualization.draw_geometries(trees, window_name="Segmented Trees")


# 提取主枝干骨架
def extract_trunk(tree_pcd, height_threshold=2.0):
    points = np.asarray(tree_pcd.points)
    # 根据高度分割点云
    trunk_points = points[points[:, 2] < height_threshold]
    trunk_pcd = o3d.geometry.PointCloud()
    trunk_pcd.points = o3d.utility.Vector3dVector(trunk_points)
    return trunk_pcd

# 提取第一棵树的主干
trunk_pcd = extract_trunk(trees[0])
o3d.visualization.draw_geometries([trunk_pcd], window_name="Tree Trunk")


# 基于骨架提取多级分枝
def extract_branches(tree_pcd, min_branch_length=0.5):
    points = np.asarray(tree_pcd.points)
    # 使用DBSCAN聚类提取枝干
    clustering = DBSCAN(eps=min_branch_length, min_samples=10).fit(points)
    labels = clustering.labels_
    branch_point_clouds = []
    for i in range(labels.max() + 1):
        mask = labels == i
        branch_points = points[mask]
        branch_pcd = o3d.geometry.PointCloud()
        branch_pcd.points = o3d.utility.Vector3dVector(branch_points)
        branch_point_clouds.append(branch_pcd)
    return branch_point_clouds

# 提取第一棵树的分枝
branches = extract_branches(trees[0])
for i, branch in enumerate(branches):
    branch.paint_uniform_color(np.random.rand(3))
o3d.visualization.draw_geometries(branches, window_name="Tree Branches")


# 根据密度或颜色分离枝干和叶片
def separate_branches_and_leaves(tree_pcd, density_threshold=50):
    points = np.asarray(tree_pcd.points)
    densities = tree_pcd.compute_nearest_neighbor_distance()
    mask = densities < density_threshold
    leaves_points = points[mask]
    branches_points = points[~mask]
    
    leaves_pcd = o3d.geometry.PointCloud()
    leaves_pcd.points = o3d.utility.Vector3dVector(leaves_points)
    
    branches_pcd = o3d.geometry.PointCloud()
    branches_pcd.points = o3d.utility.Vector3dVector(branches_points)
    
    return branches_pcd, leaves_pcd

branches_pcd, leaves_pcd = separate_branches_and_leaves(trees[0])
branches_pcd.paint_uniform_color([1, 0, 0])  # 红色表示枝干
leaves_pcd.paint_uniform_color([0, 1, 0])    # 绿色表示叶片

o3d.visualization.draw_geometries([branches_pcd, leaves_pcd], window_name="Branches and Leaves")



# 树体建模：将枝干、分枝、叶片、花、根系整合为结构化模型
def build_tree_model(trunk, branches, leaves):
    tree_model = o3d.geometry.PointCloud()
    tree_model += trunk
    for branch in branches:
        tree_model += branch
    tree_model += leaves
    return tree_model

tree_model = build_tree_model(trunk_pcd, branches, leaves_pcd)
o3d.visualization.draw_geometries([tree_model], window_name="Tree Model")