import numpy as np

class RiskEvaluator:
    """
    Fast vectorized hazard and collision risk evaluator.
    Computes danger proximity for dynamic vehicles (Class 4) and static obstacles (Class 3).
    """
    def __init__(self, high_risk_classes=(3, 4), max_danger_radius=25.0):
        self.high_risk_classes = high_risk_classes
        self.max_danger_radius = max_danger_radius

    def evaluate(self, points: np.ndarray, labels: np.ndarray) -> dict:
        """
        Vectorized evaluation. Returns max risk score [0.0 - 1.0], hazardous object count,
        and closest obstacle distance in meters.
        """
        if len(points) == 0:
            return {'max_risk': 0.0, 'hazard_count': 0, 'closest_obstacle_m': 100.0}
            
        mask = np.isin(labels, self.high_risk_classes)
        if not np.any(mask):
            return {'max_risk': 0.0, 'hazard_count': 0, 'closest_obstacle_m': 100.0}
            
        danger_pts = points[mask]
        dists = np.sqrt(danger_pts[:, 0]**2 + danger_pts[:, 1]**2)
        
        min_dist = float(np.min(dists))
        risk_scores = np.clip(1.0 - (dists / self.max_danger_radius), 0.0, 1.0)
        max_risk = float(np.max(risk_scores))
        hazard_count = int(np.sum(dists < 15.0))
        
        return {
            'max_risk': round(max_risk, 2),
            'hazard_count': hazard_count,
            'closest_obstacle_m': round(min_dist, 1)
        }