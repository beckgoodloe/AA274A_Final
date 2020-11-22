def plot_line_segments(*args, **kwargs):
    return None

## Magical black-box collition checking
def ccw(A, B, C):
    return np.cross(B - A, C - A) > 0

def line_line_intersection(l1, l2):
    A, B = np.array(l1)
    C, D = np.array(l2)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
