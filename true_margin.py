'''
Using Python 3.13.1
'''
import numpy as np

def calculate_slope(p1, p2):
    """Calculate slope of line passing through two points"""
    if p2[0] - p1[0] == 0:  # Vertical line
        return float('inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def calculate_middle_point(p1, p2):
    """Calculate middle point between two points"""
    return [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2]

def point_to_line_distance(point, slope, point_on_line):
    """Calculate perpendicular distance from a point to a line"""
    if slope == float('inf'):  # Vertical line
        return abs(point[0] - point_on_line[0])
    
    # Line equation: y = mx + b
    # Perpendicular line slope: -1/m
    # Distance formula: |ax + by + c|/sqrt(a² + b²)
    # where ax + by + c = 0 is the line equation in general form
    
    # Convert to general form: y = mx + b -> mx - y + b = 0
    b = point_on_line[1] - slope * point_on_line[0]
    
    # Calculate distance
    numerator = abs(slope * point[0] - point[1] + b)
    denominator = np.sqrt(slope**2 + 1)
    
    return numerator/denominator

def check_classification(points, labels, slope, middle_point):
    """Check if all points are correctly classified by the line"""
    # Get perpendicular slope
    perp_slope = -1/slope if slope != 0 else float('inf')
    
    # For each point, check if it's on the correct side of the line
    for i, point in enumerate(points):
        # Calculate signed distance to determine side of line
        if slope == float('inf'):
            signed_distance = point[0] - middle_point[0]
        else:
            b = middle_point[1] - slope * middle_point[0]
            signed_distance = slope * point[0] - point[1] + b
            
        # Check if classification is correct
        if (signed_distance > 0 and labels[i] < 0) or (signed_distance < 0 and labels[i] > 0):
            return False
    return True

def calculate_true_margin(X, y):
    """Calculate the true margin for the dataset"""
    max_margin = 0
    n_samples = len(X)
    
    # For each combination of three points
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            for k in range(n_samples):
                # Skip if k is i or j
                if k == i or k == j:
                    continue
                    
                # Get points
                p1, p2, p3 = X[i], X[j], X[k]
                
                # Skip if p1 and p2 are not from the same class
                if y[i] != y[j]:
                    continue
                
                # Skip if p3 is from the same class as p1 and p2
                if y[k] == y[i]:
                    continue
                
                # Calculate slope of line through p1 and p2
                p1_p2_slope = calculate_slope(p1, p2)
                
                # Calculate middle points
                p1_p2_middle = calculate_middle_point(p1, p2)
                middle_point = calculate_middle_point(p1_p2_middle, p3)
                
                # Check if this line correctly classifies all points
                if not check_classification(X, y, p1_p2_slope, middle_point):
                    continue
                
                # Calculate minimum distance from all points to the line
                min_distance = float('inf')
                for point in X:
                    distance = point_to_line_distance(point, p1_p2_slope, middle_point)
                    min_distance = min(min_distance, distance)
                
                # Update max margin if this is larger
                max_margin = max(max_margin, min_distance)
    
    return max_margin 