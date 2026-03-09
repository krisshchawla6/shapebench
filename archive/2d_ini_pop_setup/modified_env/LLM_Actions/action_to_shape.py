import numpy as np
import math

# Environment constants (from parametered_env.py)
MAX_DEFORMATION = 3.0
N_CONTROL_PTS = 4

def convert_action_to_shape(action_csv_path, shape_csv_path, n_cp=N_CONTROL_PTS, max_deformation=MAX_DEFORMATION):
    """
    Converts a flat Action CSV (Nx3: radius_param, angle_param, edgy_param) to the standard Shape CSV format.
    
    Action CSV Format:
      Single row: r0, a0, e0, r1, a1, e1, ...
      All values in [-1.0, 1.0]
    
    Shape CSV Format:
      Line 1: N_control_pts N_sampling_pts
      Lines 2..N+1: Radius (bezier control radius)
      Lines N+2..2N+1: Edgy
      Lines 2N+2..: X Y (coordinates)
    
    Conversion (matching environment.py logic):
      radius = max(abs(radius_param), 0.2) * max_deformation
      dangle = 360.0 / n_cp
      angle = dangle * pt_idx + angle_param * dangle / 2.0
      x = radius * cos(angle)
      y = radius * sin(angle)
      edgy = 0.5 + 0.5 * abs(edgy_param)
    """
    try:
        # Load action
        action = np.loadtxt(action_csv_path, delimiter=',')
        if action.ndim == 2:
            action = action[0]
            
        n_vals = len(action)
        if n_vals % 3 != 0:
            raise ValueError(f"Action array length {n_vals} is not divisible by 3")
            
        n_cp = n_vals // 3
        n_sp = 10  # Default sampling
        
        # Reshape to (N, 3) -> [radius_param, angle_param, edgy_param]
        action_matrix = action.reshape((n_cp, 3))
        
        # Convert params to actual values using environment logic
        dangle = 360.0 / float(n_cp)
        
        x_coords = np.zeros(n_cp)
        y_coords = np.zeros(n_cp)
        edgy_vals = np.zeros(n_cp)
        radius_vals = np.zeros(n_cp)
        
        for i in range(n_cp):
            radius_param = action_matrix[i, 0]
            angle_param = action_matrix[i, 1]
            edgy_param = action_matrix[i, 2]
            
            # Apply environment conversion
            radius = max(abs(radius_param), 0.2) * max_deformation
            angle = dangle * float(i) + angle_param * dangle / 2.0
            x = radius * math.cos(math.radians(angle))
            y = radius * math.sin(math.radians(angle))
            edgy = 0.5 + 0.5 * abs(edgy_param)
            
            x_coords[i] = x
            y_coords[i] = y
            edgy_vals[i] = edgy
            radius_vals[i] = 0.5  # Shape CSV uses fixed bezier radius
        
        with open(shape_csv_path, 'w') as f:
            # Header
            f.write(f"{n_cp} {n_sp}\n")
            
            # Radii (lines 2 to n_cp+1) - bezier control radius, typically 0.5
            for r in radius_vals:
                f.write(f"{r:.6f}\n")
                
            # Edgy (lines n_cp+2 to 2*n_cp+1)
            for e in edgy_vals:
                f.write(f"{e:.6f}\n")
                
            # Coordinates (lines 2*n_cp+2 onwards)
            for i in range(n_cp):
                f.write(f"{x_coords[i]:.6f} {y_coords[i]:.6f}\n")
                
        return True
        
    except Exception as e:
        print(f"Error converting action to shape: {e}")
        return False
