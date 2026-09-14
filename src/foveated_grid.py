import numpy as np
import base64

class AdaptiveFoveatedGrid:
    """
    High-performance 2.5D Adaptive Foveated Spatial Grid Engine.
    
    Resolutions based on distance from ego-vehicle:
    - Near Range (0 - 10m):   5 cm / cell   (High fidelity for safety)
    - Mid Range (10 - 30m):   15 cm / cell  (Balanced context)
    - Far Range (30 - 100m):  50 cm / cell  (Aggressive compression)
    
    Generates:
    1. 2.5D Semantic compressed cells with min/max Z and dominant risk class.
    2. 2.5D Elevation matrix for real-time turbo/jet heatmap rendering.
    3. 3D Oriented Bounding Boxes for dynamic vehicles and obstacles.
    4. Exact memory reduction analytics.
    """
    def __init__(self, x_range=(-50.0, 50.0), y_range=(-50.0, 50.0)):
        self.x_range = x_range
        self.y_range = y_range

    def process(self, points: np.ndarray, labels: np.ndarray) -> dict:
        """
        Processes raw points and semantic labels into adaptive 2.5D grid representation.
        """
        if len(points) == 0:
            return {
                'cells': [],
                'elevation_map': np.zeros((64, 64), dtype=np.float32).tolist(),
                'bounding_boxes': [],
                'raw_points_count': 0,
                'compressed_cells_count': 0,
                'compression_ratio': 0.0,
                'memory_usage_mb': 0.0,
                'raw_memory_mb': 0.0
            }

        # Filter points within bounds
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        valid_mask = (x >= self.x_range[0]) & (x <= self.x_range[1]) & \
                     (y >= self.y_range[0]) & (y <= self.y_range[1])
        
        valid_pts = points[valid_mask]
        valid_lbls = labels[valid_mask] if labels is not None else np.zeros(len(valid_pts), dtype=np.int32)
        
        N_pts = len(valid_pts)
        if N_pts == 0:
            valid_pts = points[:100]
            valid_lbls = np.zeros(len(valid_pts), dtype=np.int32)
            N_pts = len(valid_pts)

        # Distances in XY plane
        distances = np.sqrt(valid_pts[:, 0]**2 + valid_pts[:, 1]**2)
        
        # Adaptive cell resolution assignment
        cell_res = np.full(N_pts, 0.50, dtype=np.float32) # Far (30-100m)
        cell_res[distances <= 30.0] = 0.15                 # Mid (10-30m)
        cell_res[distances <= 10.0] = 0.05                 # Near (0-10m)

        # Quantize to 2D grid coordinates
        grid_x = np.floor(valid_pts[:, 0] / cell_res).astype(np.int32)
        grid_y = np.floor(valid_pts[:, 1] / cell_res).astype(np.int32)
        
        # Build 2.5D cell map using spatial hashing
        # Map key: (res_tier, grid_x, grid_y)
        res_tiers = np.where(distances <= 10.0, 0, np.where(distances <= 30.0, 1, 2))
        
        # Fully Vectorized Spatial Hashing for High FPS
        # 1. Priorities: Dynamic(4)=100, Static(3)=90, others keep original value
        priority = valid_lbls.copy()
        priority[valid_lbls == 4] = 100
        priority[valid_lbls == 3] = 90
        
        # 2. Sort by priority descending to keep the highest priority label for each bucket
        sort_idx = np.argsort(-priority)
        
        sorted_z = valid_pts[sort_idx, 2]
        sorted_lbls = valid_lbls[sort_idx]
        sorted_res = cell_res[sort_idx]
        sorted_grid_x = grid_x[sort_idx]
        sorted_grid_y = grid_y[sort_idx]
        sorted_tiers = res_tiers[sort_idx]
        
        # 3. Create unique 64-bit keys for buckets
        # tiers (0-2) in highest bits, grid_x and grid_y (approx -2000 to 2000) in lower bits
        keys = (sorted_tiers.astype(np.int64) << 40) | ((sorted_grid_x.astype(np.int64) & 0xFFFFF) << 20) | (sorted_grid_y.astype(np.int64) & 0xFFFFF)
        
        unique_keys, unique_indices, inverse_indices = np.unique(keys, return_index=True, return_inverse=True)
        
        # 4. Extract dominant features for each bucket (first occurrence has highest priority)
        final_labels = sorted_lbls[unique_indices]
        final_res = sorted_res[unique_indices]
        final_grid_x = sorted_grid_x[unique_indices]
        final_grid_y = sorted_grid_y[unique_indices]
        
        # 5. Fast Min/Max Z aggregation
        max_z = np.full(len(unique_keys), -1000.0, dtype=np.float32)
        min_z = np.full(len(unique_keys), 1000.0, dtype=np.float32)
        np.maximum.at(max_z, inverse_indices, sorted_z)
        np.minimum.at(min_z, inverse_indices, sorted_z)
        
        # 6. Calculate exactly aligned centers
        bucket_center_x = (final_grid_x * final_res) + (final_res / 2.0)
        bucket_center_y = (final_grid_y * final_res) + (final_res / 2.0)
        delta_z = max_z - min_z
        
        # 7. Fast matrix formatting for JSON
        compressed_cells_arr = np.column_stack((
            np.round(bucket_center_x, 2),
            np.round(bucket_center_y, 2),
            np.round(max_z, 2),
            np.round(delta_z, 2),
            final_labels,
            np.round(final_res, 2)
        ))
        
        compressed_cells = compressed_cells_arr.flatten().tolist()

        # Generate 2.5D Elevation Map (64x64 grid matrix normalized for turbo colormap)
        grid_dim = 64
        elev_matrix = np.full((grid_dim, grid_dim), -2.0, dtype=np.float32)
        
        x_indices = np.clip(np.floor((valid_pts[:, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * grid_dim).astype(int), 0, grid_dim - 1)
        y_indices = np.clip(np.floor((valid_pts[:, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * grid_dim).astype(int), 0, grid_dim - 1)
        
        for k in range(0, N_pts, max(1, N_pts // 8000)):
            gx, gy = x_indices[k], y_indices[k]
            val = float(valid_pts[k, 2])
            if val > elev_matrix[gy, gx]:
                elev_matrix[gy, gx] = val
                
        # Clip elevation between -2.0m and +3.0m
        elev_matrix = np.clip(elev_matrix, -2.0, 3.0)

        # Extract 3D Bounding Boxes for dynamic objects & static obstacles
        bounding_boxes = self._extract_bounding_boxes(valid_pts, valid_lbls)

        # Calculate memory & compression analytics
        raw_bytes = len(points) * 16 # 4 floats = 16 bytes per raw LiDAR point
        compressed_bytes = (len(compressed_cells) // 6) * 12 # 2.5D compact representation
        raw_mb = round(raw_bytes / (1024 * 1024), 2)
        comp_mb = round(compressed_bytes / (1024 * 1024), 2)
        comp_ratio = round((1.0 - (comp_mb / max(raw_mb, 0.001))) * 100.0, 1)

        # Spatially distributed downsampling (Voxel Grid) for raw points
        voxel_size = 0.6
        voxel_coords = np.floor(valid_pts[:, :3] / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        downsampled_raw_points = valid_pts[unique_indices, :3].flatten().tolist()

        return {
            'cells': compressed_cells,
            'elevation_map': elev_matrix.tolist(),
            'bounding_boxes': bounding_boxes,
            'raw_points': downsampled_raw_points,
            'raw_points_count': len(points),
            'compressed_cells_count': len(compressed_cells) // 6,
            'compression_ratio': comp_ratio,
            'memory_usage_mb': comp_mb,
            'raw_memory_mb': raw_mb
        }

    def _extract_bounding_boxes(self, points: np.ndarray, labels: np.ndarray) -> list:
        """
        Clusters obstacle and vehicle points to produce 3D oriented bounding boxes:
        [center_x, center_y, center_z, size_x, size_y, size_z, label_id]
        """
        boxes = []
        # Find dynamic objects (4) and static obstacles (3)
        for target_label in [4, 3]:
            mask = (labels == target_label)
            target_pts = points[mask]
            
            if len(target_pts) < 15:
                continue
                
            # Quick grid-based spatial clustering
            cluster_res = 2.5 # 2.5m cluster radius
            cluster_keys = np.floor(target_pts[:, :2] / cluster_res).astype(int)
            unique_keys = np.unique(cluster_keys, axis=0)
            
            for key in unique_keys[:8]: # Max 8 salient bounding boxes per frame
                c_mask = (cluster_keys[:, 0] == key[0]) & (cluster_keys[:, 1] == key[1])
                c_pts = target_pts[c_mask]
                
                if len(c_pts) >= 12:
                    min_coords = np.min(c_pts[:, :3], axis=0)
                    max_coords = np.max(c_pts[:, :3], axis=0)
                    center = (min_coords + max_coords) / 2.0
                    size = np.maximum(max_coords - min_coords, [1.2, 1.2, 1.0])
                    
                    boxes.append({
                        'center': [round(float(center[0]), 2), round(float(center[1]), 2), round(float(center[2]), 2)],
                        'size': [round(float(size[0]), 2), round(float(size[1]), 2), round(float(size[2]), 2)],
                        'label': int(target_label)
                    })
                    
        return boxes