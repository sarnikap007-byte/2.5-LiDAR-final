import numpy as np
import os

class KITTILoader:
    """
    Loads real-world 3D LiDAR point clouds and labels from SemanticKITTI dataset.
    Reads .bin files (float32 [x, y, z, reflectance]) and .label files (uint32 semantic IDs).
    Provides realistic autonomous vehicle simulation progression across consecutive frames.
    """
    def __init__(self, data_dir: str = "./dataset/sequences/00/velodyne"):
        self.data_dir = data_dir
        if os.path.exists(data_dir):
            self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
        else:
            self.files = []
            
        self.cached_base_points = None
        self.cached_base_labels = None

    def __len__(self):
        return max(len(self.files), 500) # Loop sequence smoothly

    def load_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads frame (points, mapped_labels).
        If dataset contains few frames or the file is truncated, applies realistic vehicle ego-motion & synthetic fallback.
        """
        actual_index = index % len(self.files) if len(self.files) > 0 else 0
        
        if len(self.files) > 0:
            file_path = os.path.join(self.data_dir, self.files[actual_index])
            scan = np.fromfile(file_path, dtype=np.float32)
            points = scan.reshape(-1, 4)[:, :4] # x, y, z, reflectance
            
            # Check if it's a truncated file (e.g. 62KB sample instead of 1.9MB full scan)
            if len(points) < 20000:
                print(f"\n[Warning] File {self.files[actual_index]} is truncated ({len(points)} pts). Switching to Dense Synthetic Data.")
                points, mapped_labels = self._generate_synthetic_urban_frame(index)
            else:
                # Load label file if present
                label_dir = self.data_dir.replace('velodyne', 'labels')
                label_file = os.path.join(label_dir, self.files[actual_index].replace('.bin', '.label'))
                
                if os.path.exists(label_file):
                    labels_raw = np.fromfile(label_file, dtype=np.uint32)
                    semantic_ids = labels_raw & 0xFFFF
                    mapped_labels = self._map_kitti_classes(semantic_ids)
                else:
                    mapped_labels = self._generate_geometric_labels(points)
        else:
            points, mapped_labels = self._generate_synthetic_urban_frame(index)

        # Apply realistic ego-motion and dynamic traffic movement for continuous playback
        if len(self.files) <= 5:
            points, mapped_labels = self._apply_dynamic_simulation(points, mapped_labels, index)

        return points, mapped_labels

    def _map_kitti_classes(self, semantic_ids: np.ndarray) -> np.ndarray:
        """
        Map SemanticKITTI official class IDs into the 7 dashboard classes:
        1: Drivable Terrain (Road, parking)
        2: Non-Drivable Terrain (Sidewalk, terrain)
        3: Static Obstacle (Pole, traffic sign, fence, trunk)
        4: Dynamic Object (Car, truck, bicycle, person, motorcyclist)
        5: Vegetation (Trees, foliage)
        6: Building (Building, wall)
        """
        mapped = np.zeros_like(semantic_ids, dtype=np.int32)
        
        # 1: Drivable Terrain (Road=40, Parking=44, Lane-marking=48, Other-ground=49)
        mapped[(semantic_ids == 40) | (semantic_ids == 44) | (semantic_ids == 48) | (semantic_ids == 49)] = 1
        
        # 2: Non-Drivable Terrain (Sidewalk=48, Terrain=72)
        mapped[(semantic_ids == 48) | (semantic_ids == 72)] = 2
        
        # 3: Static Obstacle (Fence=51, Pole=80, Traffic-sign=81)
        mapped[(semantic_ids == 51) | (semantic_ids == 80) | (semantic_ids == 81)] = 3
        
        # 4: Dynamic Object (Car=10, Bicycle=11, Motorcycle=13, Truck=18, Other-vehicle=20, Person=30, Bicyclist=31, Motorcyclist=32, + 250-259)
        dynamic_mask = (semantic_ids >= 10) & (semantic_ids <= 39) | (semantic_ids >= 250) & (semantic_ids <= 259)
        mapped[dynamic_mask] = 4
        
        # 5: Vegetation (Vegetation=70, Trunk=71)
        mapped[(semantic_ids == 70) | (semantic_ids == 71)] = 5
        
        # 6: Building (Building=50, Other-structure=52)
        mapped[(semantic_ids == 50) | (semantic_ids == 52)] = 6
        
        return mapped

    def _generate_geometric_labels(self, points: np.ndarray) -> np.ndarray:
        """Rule-based segmentation for unlabelled raw scans"""
        N = len(points)
        labels = np.zeros(N, dtype=np.int32)
        z = points[:, 2]
        x = points[:, 0]
        y = points[:, 1]
        dist = np.sqrt(x**2 + y**2)
        
        # Drivable road
        is_road = (z >= -2.2) & (z <= -1.4) & (np.abs(y) <= 3.8) & (dist < 55.0)
        labels[is_road] = 1
        
        # Non-drivable sidewalk/curb
        is_curb = (z >= -2.2) & (z <= -1.3) & (np.abs(y) > 3.8) & (np.abs(y) <= 7.0)
        labels[is_curb] = 2
        
        # Dynamic vehicles (clusters near road)
        is_veh1 = (z > -1.4) & (z < 0.6) & (np.abs(x - 12.0) < 2.5) & (np.abs(y - 1.8) < 1.4)
        is_veh2 = (z > -1.4) & (z < 0.6) & (np.abs(x + 15.0) < 2.5) & (np.abs(y + 2.0) < 1.4)
        is_veh3 = (z > -1.4) & (z < 0.6) & (np.abs(x - 24.0) < 2.5) & (np.abs(y + 1.8) < 1.4)
        labels[is_veh1 | is_veh2 | is_veh3] = 4
        
        # Static obstacles
        is_static = (z > -1.4) & (z < 0.5) & (np.abs(y) >= 4.0) & (np.abs(y) <= 5.5) & (dist < 40.0)
        labels[is_static] = 3
        
        # Vegetation
        is_veg = (z > 0.0) & (np.abs(y) >= 5.0) & (dist < 60.0)
        labels[is_veg] = 5
        
        # Buildings
        is_bld = (z > 1.0) & (np.abs(y) >= 12.0)
        labels[is_bld] = 6
        
        return labels

    def _generate_synthetic_urban_frame(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates realistic dense LiDAR urban intersection point cloud that forms a SOLID gapless 2.5D map"""
        points_list = []
        labels_list = []
        
        # 1. Multi-Resolution Ground Mesh (Perfectly matching foveated grid)
        # We generate 3 zones to ensure solid gapless mapping without massive point counts.
        
        # Zone A (Near): 0.05m spacing, 15m range
        rx_a, ry_a = np.meshgrid(np.arange(-15.0, 15.0, 0.05), np.arange(-15.0, 15.0, 0.05))
        
        # Zone B (Mid): 0.15m spacing, 35m range (exclude inner 15m)
        rx_b, ry_b = np.meshgrid(np.arange(-35.0, 35.0, 0.15), np.arange(-35.0, 35.0, 0.15))
        mid_mask = (np.abs(rx_b) >= 15.0) | (np.abs(ry_b) >= 15.0)
        rx_b, ry_b = rx_b[mid_mask], ry_b[mid_mask]
        
        # Zone C (Far): 0.50m spacing, 45m range (exclude inner 35m)
        rx_c, ry_c = np.meshgrid(np.arange(-45.0, 45.0, 0.50), np.arange(-45.0, 45.0, 0.50))
        far_mask = (np.abs(rx_c) >= 35.0) | (np.abs(ry_c) >= 35.0)
        rx_c, ry_c = rx_c[far_mask], ry_c[far_mask]
        
        rx = np.concatenate([rx_a.flatten(), rx_b.flatten(), rx_c.flatten()])
        ry = np.concatenate([ry_a.flatten(), ry_b.flatten(), ry_c.flatten()])
        
        # Add slight noise to z for realism
        rz = -1.73 + np.random.normal(0, 0.005, len(rx))
        
        # Determine road vs non-road (Cross intersection)
        is_road = (np.abs(ry) <= 4.5) | (np.abs(rx) <= 4.5)
        dist = np.sqrt(rx**2 + ry**2)
        is_road = is_road | (dist < 12.0)
        
        r_refl = np.clip(0.3 + 0.5 * np.exp(-dist / 20.0), 0.1, 0.9)
        
        # Road (Class 1)
        points_list.append(np.column_stack([rx[is_road], ry[is_road], rz[is_road], r_refl[is_road]]))
        labels_list.append(np.full(np.sum(is_road), 1, dtype=np.int32))
        
        # Non-Drivable Terrain / Vegetation blocks (Class 2 and 5)
        non_road_mask = ~is_road
        nx, ny, nz = rx[non_road_mask], ry[non_road_mask], rz[non_road_mask]
        
        is_veg = (nx > 10) & (ny > 10) | (nx < -10) & (ny < -10)
        labels_non_road = np.where(is_veg, 5, 2)
        points_list.append(np.column_stack([nx, ny, nz, r_refl[non_road_mask]]))
        labels_list.append(labels_non_road.astype(np.int32))
        
        # 2. Solid Buildings (Class 6)
        # Create solid blocks for buildings
        for bx, by, bw, bh in [(-25, -25, 10, 10), (15, -25, 20, 10), (-30, 15, 15, 20), (25, 25, 15, 15)]:
            cx, cy = np.meshgrid(np.arange(bx, bx+bw, 0.2), np.arange(by, by+bh, 0.2))
            cx, cy = cx.flatten(), cy.flatten()
            # extrude vertically
            cz = np.random.uniform(-1.7, 6.0, len(cx)*5) # 5 points per grid cell vertically
            cx = np.repeat(cx, 5)
            cy = np.repeat(cy, 5)
            points_list.append(np.column_stack([cx, cy, cz, np.full(len(cx), 0.8)]))
            labels_list.append(np.full(len(cx), 6, dtype=np.int32))
            
        # 3. Dynamic Vehicles (Class 4) - Solid boxes
        car_offset = (frame_idx * 0.5) % 60 - 30
        for v_x, v_y, l, w in [(12 + car_offset, 2.0, 4.0, 2.0), (-18 - car_offset, -2.0, 4.0, 2.0), (2.0, 15 + car_offset, 2.0, 4.0)]:
            vx, vy = np.meshgrid(np.arange(v_x - l/2, v_x + l/2, 0.1), np.arange(v_y - w/2, v_y + w/2, 0.1))
            vx, vy = vx.flatten(), vy.flatten()
            vz = np.random.uniform(-1.7, 0.5, len(vx)*4)
            vx = np.repeat(vx, 4)
            vy = np.repeat(vy, 4)
            points_list.append(np.column_stack([vx, vy, vz, np.full(len(vx), 0.9)]))
            labels_list.append(np.full(len(vx), 4, dtype=np.int32))
            
        points = np.vstack(points_list).astype(np.float32)
        labels = np.concatenate(labels_list).astype(np.int32)
        return points, labels

    def _apply_dynamic_simulation(self, points: np.ndarray, labels: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Adds natural motion to vehicles and scans across frames"""
        pts_copy = points.copy()
        dyn_mask = (labels == 4)
        
        if np.any(dyn_mask):
            shift = (np.sin(frame_idx * 0.15) * 4.0)
            pts_copy[dyn_mask, 0] += shift
            
        return pts_copy, labels