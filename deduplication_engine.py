import math
import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple
from torchvision import transforms

class DeduplicationEngine:
    def __init__(self, geo_radius_meters: float = 75.0, inlier_threshold_merge: int = 35, inlier_threshold_review: int = 15):
        self.geo_radius = geo_radius_meters
        self.inlier_merge = inlier_threshold_merge
        self.inlier_review = inlier_threshold_review
        
        # Load DINOv2 ViT-S/14 (Small, fast on CPU ~21M params, 384-dim embedding)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dinov2.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # SIFT feature extractor for geometric verification fallback / local testing
        self.sift = cv2.SIFT_create()

    # --- Stage 0: Haversine Geo-Fence ---
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    # --- Stage 1: DINOv2 Global Instance Embedding ---
    def get_dinov2_embedding(self, img: Image.Image) -> np.ndarray:
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.dinov2(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy()

    # --- Stage 2: Geometric Verification (RANSAC Feature Inliers) ---
    def count_geometric_inliers(self, img1_path: str, img2_path: str) -> int:
        im1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            return 0

        kp1, des1 = self.sift.detectAndCompute(im1, None)
        kp2, des2 = self.sift.detectAndCompute(im2, None)
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return 0

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return len(good)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC homography estimation to verify real geometric alignment
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return int(np.sum(mask)) if mask is not None else 0

    # --- Pipeline Orchestrator & 3-Band Decision ---
    def evaluate_incoming_report(
        self, 
        new_image_path: str, 
        new_lat: float, 
        new_lng: float, 
        existing_tickets: List[Dict]
    ) -> Dict:
        """
        existing_tickets: list of dicts with keys: ['id', 'lat', 'lng', 'image_url', 'local_image_path', 'embedding', 'root_ticket_id']
        """
        img = Image.open(new_image_path).convert('RGB')
        new_emb = self.get_dinov2_embedding(img)

        # Stage 0: Geo-Filter
        candidates = [
            t for t in existing_tickets 
            if self.haversine_distance(new_lat, new_lng, t['lat'], t['lng']) <= self.geo_radius
        ]

        if not candidates:
            return {"action": "CREATE_NEW", "embedding": new_emb.tolist(), "matched_ticket_id": None, "inliers": 0}

        # Stage 1: DINOv2 Cosine Similarity Ranking
        ranked_candidates = []
        for c in candidates:
            c_emb = np.array(c['embedding'])
            sim = float(np.dot(new_emb, c_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(c_emb)))
            ranked_candidates.append((sim, c))

        ranked_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = ranked_candidates[:5]

        # Stage 2: Geometric Verification on Top Candidates
        best_inliers = 0
        best_candidate = None

        for sim, cand in top_candidates:
            inliers = self.count_geometric_inliers(new_image_path, cand['local_image_path'])
            if inliers > best_inliers:
                best_inliers = inliers
                best_candidate = cand

        # Banded Decision Logic
        if best_candidate and best_inliers >= self.inlier_merge:
            # Band 1: High Inlier Count -> Auto-Merge (resolve to root ticket)
            root_id = best_candidate.get('root_ticket_id') or best_candidate['id']
            return {
                "action": "AUTO_MERGE",
                "embedding": new_emb.tolist(),
                "matched_ticket_id": best_candidate['id'],
                "root_ticket_id": root_id,
                "inliers": best_inliers
            }
        elif best_candidate and best_inliers >= self.inlier_review:
            # Band 2: Ambiguous Region -> Send to Control Room Review Queue
            root_id = best_candidate.get('root_ticket_id') or best_candidate['id']
            return {
                "action": "FLAG_FOR_REVIEW",
                "embedding": new_emb.tolist(),
                "matched_ticket_id": best_candidate['id'],
                "root_ticket_id": root_id,
                "inliers": best_inliers
            }
        else:
            # Band 3: Low inliers -> Distinct physical issue
            return {
                "action": "CREATE_NEW",
                "embedding": new_emb.tolist(),
                "matched_ticket_id": None,
                "root_ticket_id": None,
                "inliers": best_inliers
            }