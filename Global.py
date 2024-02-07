import numpy as np
import argparse
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# Constants (Define these based on your requirements)
CYLINDER_SIZE = 100  
NS, ND = 10, 10  # number of spatial and directional cells
OMEGA = 1.0  # convex hull parameter
MAX_ANG_DIFF = 30  # Maximum angular difference
NP_MIN, NP_MAX = 5, 15  # Minimum and maximum points
SIGMA_S = 1.0  # Spatial Gaussian sigma
MICRO_PSI = 0.5  # Threshold for cylinder values
MIN_VALID_CELLS = 0.5  # Minimum valid cells threshold
MIN_CONTRIBUTING_MINUTIAE = 2  # Minimum contributing minutiae
CELL_SIZE = 1.0  

# Helper Functions
def convex_hull(minutiae):
    points = np.array(minutiae)
    hull = ConvexHull(points)
    return hull.vertices

def cell_angle(k):
    return 2 * np.pi * k / ND

def cell_center(m, i, j, sin, cos):
    x_offset = i * CELL_SIZE * cos - j * CELL_SIZE * sin
    y_offset = i * CELL_SIZE * sin + j * CELL_SIZE * cos
    return m.x + x_offset, m.y + y_offset

def ed_to_point(m, center):
    return np.sqrt((m.x - center[0])**2 + (m.y - center[1])**2)

def is_point_inside_chull(chull, center, omega):
    hull_path = Path([chull.vertices[i] for i in chull.vertices])
    return hull_path.contains_point(center)

def gaussian(dist):
    return np.exp(-0.5 * (dist / SIGMA_S) ** 2)

def ang_diff(arc1, arc2):
    diff = np.abs(arc1 - arc2)
    return min(diff, 2 * np.pi - diff)

def area_under_gaussian(ang_diff):
    return gaussian(ang_diff) * np.sqrt(2 * np.pi) * SIGMA_S

def linearize_idxs(i, j, k):
    return i * NS * ND + j * ND + k

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

class LinearizedBitCylinder:
    def __init__(self, data=None):
        self.data = data if data is not None else [False] * CYLINDER_SIZE

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __and__(self, other):
        return LinearizedBitCylinder([a and b for a, b in zip(self.data, other.data)])

    def __or__(self, other):
        return LinearizedBitCylinder([a or b for a, b in zip(self.data, other.data)])

    def __xor__(self, other):
        return LinearizedBitCylinder([a != b for a, b in zip(self.data, other.data)])

    def __invert__(self):
        return LinearizedBitCylinder([not a for a in self.data])

    def count(self):
        return sum(self.data)


class BitCylinderSet:
    def __init__(self, minutiae, additional_weights=None):
        self.cylinders = []
        self.validities = []
        self.minutiae_references = []
       
        chull = convex_hull(minutiae)
        phi_vals = [cell_angle(k) for k in range(1, ND+1)]

        for idx_probe in range(len(minutiae)):
            m = minutiae[idx_probe]
            tmp_cylinder = [0.0] * CYLINDER_SIZE
           
            cyl_values = LinearizedBitCylinder()
            cyl_validity = LinearizedBitCylinder()
            cyl_validity = ~cyl_validity
           
            sin, cos = np.sin(m.arc()), np.cos(m.arc())
           
            contributing_minutiae = [0] * len(minutiae)
            invalid_cells = 0
           
            for i in range(NS):
                for j in range(NS):
                    center = cell_center(m, i + 1, j + 1, sin, cos)
                   
                    if ed_to_point(m, center) > CYLINDER_RADIUS or not is_point_inside_chull(chull, center, OMEGA):
                        for k in range(ND):
                            cyl_validity[linearize_idxs(i, j, k)] = False
                        invalid_cells += 1
                        continue
                   
                    for idx_cand in range(len(minutiae)):
                        if idx_probe == idx_cand:
                            continue
                       
                        mt = minutiae[idx_cand]
                        dist = ed_to_point(mt, center)
                        if dist > 3.0 * SIGMA_S:
                            continue
                        spatial_contribution = gaussian(dist)
                        contributing_minutiae[idx_cand] = 1
                        minutiae_ang_diff = ang_diff(m.arc(), mt.arc())
                        for k in range(ND):
                            d_phi = phi_vals[k]
                            ang_difference = ang_diff(d_phi, minutiae_ang_diff)
                            directional_contribution = area_under_gaussian(ang_difference)

                            # Apply additional weights if provided
                            weight = additional_weights.get(idx_cand, 1.0) if additional_weights else 1.0
                            tmp_cylinder[linearize_idxs(i, j, k)] += weight * spatial_contribution * directional_contribution
           
            sum_of_contributing = sum(contributing_minutiae)
            if (1.0 - invalid_cells / (NS * NS) < MIN_VALID_CELLS) or (sum_of_contributing < MIN_CONTRIBUTING_MINUTIAE):
                continue
           
            for i in range(CYLINDER_SIZE):
                if tmp_cylinder[i] > MICRO_PSI:
                    cyl_values[i] = True
            self.cylinders.append(cyl_values)
            self.validities.append(cyl_validity)
            self.minutiae_references.append(idx_probe)

    def len(self):
        return len(self.cylinders)

def bitwise_sum(b):
    return b.count()

def bitwise_norm(b):
    return np.sqrt(b.count())

def calc_nP(c1, c2):
    return NP_MIN + (NP_MAX - NP_MIN) * sigmoid(min(c1.len(), c2.len()), 20.0, 2.0 / 5.0) + 0.5

def local_matching(c1, m1, c2, m2):
    similarities = []
   
    for i in range(c1.len()):
        for j in range(c2.len()):
            if ang_diff(m1[c1.minutiae_references[i]].arc(), m2[c2.minutiae_references[j]].arc()) > MAX_ANG_DIFF:
                similarities.append(0.0)
                continue
            validitiy_ab = c1.validities[i] & c2.validities[j]
            c_ab = c1.cylinders[i] & validitiy_ab
            c_ba = c2.cylinders[j] & validitiy_ab
           
            if bitwise_sum(validitiy_ab) / CYLINDER_SIZE < 0.6:
                similarities.append(0.0)
                continue
            norm_c_ab = bitwise_norm(c_ab)
            norm_c_ba = bitwise_norm(c_ba)
           
            if norm_c_ab + norm_c_ba == 0:
                similarities.append(0.0)
                continue
            cxor = c_ab ^ c_ba
            lambda_ = 1.0 - (bitwise_norm(cxor) / (norm_c_ab + norm_c_ba))
            similarities.append(lambda_)
   
    return similarities

def global_matching(similarities, nP):
    similarities = np.array(similarities)
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = similarities[sorted_indices]
    sum_ = np.sum(sorted_similarities[:nP])
    return sum_ / nP

def global_matching_LSA(similarities, c1, c2):
    nP = calc_nP(c1, c2)
    matrix = np.array(similarities).reshape(c1.len(), c2.len())
    macopy = np.copy(matrix)
   
    for i in range(c1.len()):
        for j in range(c2.len()):
            if matrix[i, j] <= 0.0:
                matrix[i, j] = 999
            else:
                matrix[i, j] = 1.0 / matrix[i, j]
   
    row_ind, col_ind = linear_sum_assignment(matrix)
    valid_scores = []
    for i in range(c1.len()):
        for j in range(c2.len()):
            if matrix[i, j] == 0:
                valid_scores.append(macopy[i, j])
    valid_scores = np.array(valid_scores)
    sorted_indices = np.argsort(valid_scores)[::-1]
    sorted_valid_scores = valid_scores[sorted_indices]
    sum_ = np.sum(sorted_valid_scores[:nP])
    return sum_ / nP

minutiae = [...]  # Your minutiae data
additional_weights = {0: 1.5, 1: 1.2}  # Additional weights
bit_cylinder_set1 = BitCylinderSet(minutiae, additional_weights)  
bit_cylinder_set2 = BitCylinderSet(minutiae, additional_weights)  

# Calculate local matching
local_similarities = local_matching(bit_cylinder_set1, minutiae, bit_cylinder_set2, minutiae)

# Calculate global matching
nP = calc_nP(bit_cylinder_set1, bit_cylinder_set2)
global_match_score = global_matching(local_similarities, nP)
global_match_score_LSA = global_matching_LSA(local_similarities, bit_cylinder_set1, bit_cylinder_set2)