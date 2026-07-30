import os
import math
import json
import numpy as np

try:
    import quadpy
except ImportError:
    raise ImportError("The 'quadpy' library is strictly required to generate Lebedev quadrature grids. Execute 'pip install quadpy'.")

import cst.interface
from cst_debugging_tools.watchdog import CSTWatchdog
from cst_debugging_tools.safe_executor import SafeExecutor

def calculate_multipole_parameters(w, h, f_max):
    """
    Derives the dimensionless size parameter and required multipole truncation 
    limits based on the physical geometry and maximum operational frequency.
    """
    c = 299792458.0  # Speed of light in vacuum (m/s)
    
    # Calculate circumscribing sphere radius (a) in three dimensions
    a = 0.5 * math.sqrt(2 * w**2 + h**2)
    
    # Calculate maximum wavenumber (k_max)
    k_max = (2 * math.pi * f_max) / c
    
    # Calculate dimensionless size parameter (x_max)
    x_max = k_max * a
    
    # Apply Wiscombe's rigorous empirical bounds for L_max
    if 0.02 <= x_max <= 8.0:
        l_max = math.ceil(x_max + 4 * (x_max**(1/3)) + 1)
    elif 8.0 < x_max < 4200.0:
        l_max = math.ceil(x_max + 4.05 * (x_max**(1/3)) + 2)
    elif x_max >= 4200.0:
        l_max = math.ceil(x_max + 4 * (x_max**(1/3)) + 2)
    else:
        raise ValueError("x_max falls below the valid limit for standard multipole expansion.")
        
    # Determine the minimum algebraic degree of precision for Lebedev quadrature
    p_min = 2 * l_max
    
    return a, x_max, l_max, p_min

def generate_lebedev_nodes(precision):
    """
    Retrieves Lebedev quadrature nodes for the specified algebraic degree of precision
    utilizing the quadpy numerical integration library. Transforms the standard 
    Cartesian coordinates on the unit sphere to zenith (theta) and azimuth (phi) angles.
    """
    print(f"Generating Lebedev grid for minimum algebraic precision p = {precision}")
    
    # Identify the appropriate Lebedev scheme that satisfies or exceeds p_min
    # quadpy supports defined discrete precisions; we iterate to find the lowest valid scheme.
    valid_precisions = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35]
    selected_p = next((p for p in valid_precisions if p >= precision), None)
    
    if selected_p is None:
        raise ValueError(f"Required precision {precision} exceeds standard Lebedev schemes available in quadpy.")
        
    # Retrieve the Lebedev scheme geometry
    scheme = quadpy.u3.lebedev(selected_p)
    points_cartesian = scheme.points
    
    nodes = []
    for point in points_cartesian:
        x, y, z = point
        # Calculate spherical coordinates (r=1)
        # theta in [0, pi], phi in [0, 2pi]
        theta_rad = math.acos(np.clip(z, -1.0, 1.0))
        phi_rad = math.atan2(y, x)
        if phi_rad < 0:
            phi_rad += 2 * math.pi
            
        nodes.append((math.degrees(theta_rad), math.degrees(phi_rad)))
        
    print(f"Lebedev grid established with {len(nodes)} discrete nodal points.")
    return nodes

def automate_t_matrix_extraction(cst_file_path, output_directory, w, h, f_max):
    """
    Controls CST Microwave Studio to execute the Frequency Domain Solver 
    for discrete plane wave excitations and exports the complex near-field data.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 1. Analytically derive the required quadrature precision
    a, x_max, l_max, p_min = calculate_multipole_parameters(w, h, f_max)
    print(f"Calculated Parameters:")
    print(f"Circumscribing Radius (a): {a:.6e} m")
    print(f"Size Parameter (x_max): {x_max:.4f}")
    print(f"Truncation Limit (L_max): {l_max}")
    print(f"Required Quadrature Precision (p): {p_min}")

    # 2. Initialize the automated guardrail architecture
    watchdog = CSTWatchdog(timeout=30.0)
    executor = SafeExecutor(retries=2, retry_delay=3.0, default_timeout=120.0)
    
    de = watchdog.ensure_healthy_session()
    if not de:
        raise RuntimeError("Fatal execution error: Failed to establish a healthy CST process via COM.")
        
    project = de.open_project(cst_file_path)
    model = project.model3d

    # 3. Retrieve Lebedev quadrature grid
    lebedev_nodes = generate_lebedev_nodes(p_min)
    polarizations = ['TE', 'TM']
    
    # 4. Define high-density Cartesian export resolution (50 discrete segments per unit-cell width)
    spatial_step = w / 50.0
    f_max_ghz = f_max / 1e9

    metadata_manifest = []

    # 5. Execute simulation matrix
    for idx, (theta, phi) in enumerate(lebedev_nodes):
        for pol in polarizations:
            
            # Calculates the Cartesian vector components for the specified spherical angles
            theta_rad = math.radians(theta)
            phi_rad = math.radians(phi)
            
            kx = math.sin(theta_rad) * math.cos(phi_rad)
            ky = math.sin(theta_rad) * math.sin(phi_rad)
            kz = math.cos(theta_rad)

            # Define orthogonal electric field vector based on polarization state
            if pol == 'TE':
                ex, ey, ez = -math.sin(phi_rad), math.cos(phi_rad), 0.0
            else:
                ex, ey, ez = math.cos(theta_rad)*math.cos(phi_rad), math.cos(theta_rad)*math.sin(phi_rad), -math.sin(theta_rad)

            # Define the incident plane wave excitation
            vba_plane_wave = f"""
            With PlaneWave
                .Reset
                .Normal "{kx}", "{ky}", "{kz}"
                .ElectricVector "{ex}", "{ey}", "{ez}"
                .Polarization "Linear"
                .ReferenceFrequency "Center"
                .Store
            End With
            """
            executor.execute(f"Excitation Node {idx} {pol}", lambda: model.add_to_history(f"Set Excitation Node {idx} {pol}", vba_plane_wave))

            # Execute the Frequency Domain Solver utilizing the Finite Element Method (FEM)
            vba_solver = """
            With FDSolver
                .Reset
                .Start
            End With
            """
            executor.execute(f"FEM Solver Node {idx} {pol}", lambda: model.add_to_history(f"Run FEM Solver Node {idx} {pol}", vba_solver), timeout=300.0)

            # Define strictly formatted output paths for export macros
            e_export_filename = os.path.join(output_directory, f"E_Field_Node_{idx}_{pol}.txt").replace("\\", "\\\\")
            h_export_filename = os.path.join(output_directory, f"H_Field_Node_{idx}_{pol}.txt").replace("\\", "\\\\")
            
            # Select and export the complex Electric Field
            vba_export_e = f"""
            SelectTreeItem("2D/3D Results\\E-Field\\e-field (f={f_max_ghz}) [1]")
            With ASCIIExport
                .Reset
                .FileName "{e_export_filename}"
                .Mode "FixedStep"
                .StepX "{spatial_step}"
                .StepY "{spatial_step}"
                .StepZ "{spatial_step}"
                .Execute
            End With
            """
            executor.execute(f"Export E-Field {idx} {pol}", lambda: model.add_to_history(f"Export E-Field Node {idx} {pol}", vba_export_e))

            # Select and export the complex Magnetic Field
            vba_export_h = f"""
            SelectTreeItem("2D/3D Results\\H-Field\\h-field (f={f_max_ghz}) [1]")
            With ASCIIExport
                .Reset
                .FileName "{h_export_filename}"
                .Mode "FixedStep"
                .StepX "{spatial_step}"
                .StepY "{spatial_step}"
                .StepZ "{spatial_step}"
                .Execute
            End With
            """
            executor.execute(f"Export H-Field {idx} {pol}", lambda: model.add_to_history(f"Export H-Field Node {idx} {pol}", vba_export_h))

            # Formalize configuration variables for analytical post-processing ingestion
            metadata_manifest.append({
                "idx": idx,
                "theta_inc": theta,
                "phi_inc": phi,
                "polarization": pol,
                "e_file": e_export_filename,
                "h_file": h_export_filename
            })

    project.save()
    project.close()
    
    # Serialize strictly structured operational parameters
    metadata_filepath = os.path.join(output_directory, "excitation_metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata_manifest, f, indent=4)
        
    print("Computational extraction sequence complete.")

if __name__ == "__main__":
    CST_TEMPLATE_PATH = r"C:\Research\Metamaterials\UnitCell_Template.cst"
    OUTPUT_DIR = r"C:\Research\Metamaterials\Exported_Fields"
    
    # Input spatial parameters (meters) and maximum frequency (Hz)
    UNIT_CELL_WIDTH = 15e-3   # 15 mm
    UNIT_CELL_HEIGHT = 2e-3   # 2 mm
    MAX_FREQUENCY = 10e9      # 10 GHz
    
    automate_t_matrix_extraction(
        CST_TEMPLATE_PATH, 
        OUTPUT_DIR, 
        UNIT_CELL_WIDTH, 
        UNIT_CELL_HEIGHT, 
        MAX_FREQUENCY
    )