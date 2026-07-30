import os
import math
import json
import numpy as np
import scipy.special as sp
from scipy.interpolate import RegularGridInterpolator

def generate_gauss_legendre_grid(l_max):
    """
    Generates the optimal Gauss-Legendre quadrature grid for a sphere.
    Guarantees exact integration for spherical harmonics up to degree l_max.
    """
    n_theta = l_max + 1
    n_phi = 2 * l_max + 1
    
    roots, weights_theta = sp.roots_legendre(n_theta)
    theta_grid = np.arccos(roots)
    
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    weights_phi = (2 * np.pi) / n_phi
    
    return theta_grid, phi_grid, weights_theta, weights_phi

def parse_cst_ascii_export(filepath):
    """
    Parses the structured 9-column ASCII output from CST Microwave Studio.
    Format: X, Y, Z, Re(Vx), Im(Vx), Re(Vy), Im(Vy), Re(Vz), Im(Vz)
    """
    data = np.loadtxt(filepath, comments=['#', '//', '*', 'x', 'y', 'z'])
    
    x_axis = np.unique(data[:, 0])
    y_axis = np.unique(data[:, 1])
    z_axis = np.unique(data[:, 2])
    
    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    
    V_x = np.zeros((nx, ny, nz), dtype=complex)
    V_y = np.zeros((nx, ny, nz), dtype=complex)
    V_z = np.zeros((nx, ny, nz), dtype=complex)
    
    x_idx = np.searchsorted(x_axis, data[:, 0])
    y_idx = np.searchsorted(y_axis, data[:, 1])
    z_idx = np.searchsorted(z_axis, data[:, 2])
    
    V_x[x_idx, y_idx, z_idx] = data[:, 3] + 1j * data[:, 4]
    V_y[x_idx, y_idx, z_idx] = data[:, 5] + 1j * data[:, 6]
    V_z[x_idx, y_idx, z_idx] = data[:, 7] + 1j * data[:, 8]
    
    return x_axis, y_axis, z_axis, (V_x, V_y, V_z)

def build_cartesian_interpolators(x_axis, y_axis, z_axis, vector_field_data):
    """
    Constructs trilinear interpolation objects for 3D complex spatial data.
    """
    F_x, F_y, F_z = vector_field_data
    
    interp_x = RegularGridInterpolator((x_axis, y_axis, z_axis), F_x, method='linear', bounds_error=False, fill_value=0.0j)
    interp_y = RegularGridInterpolator((x_axis, y_axis, z_axis), F_y, method='linear', bounds_error=False, fill_value=0.0j)
    interp_z = RegularGridInterpolator((x_axis, y_axis, z_axis), F_z, method='linear', bounds_error=False, fill_value=0.0j)
    
    return (interp_x, interp_y, interp_z)

def interpolate_to_spherical_boundary(interpolators, r_monitor, theta_grid, phi_grid):
    """
    Evaluates the Cartesian trilinear interpolators at the specified Gauss-Legendre coordinates.
    """
    Phi, Theta = np.meshgrid(phi_grid, theta_grid)
    
    X = r_monitor * np.sin(Theta) * np.cos(Phi)
    Y = r_monitor * np.sin(Theta) * np.sin(Phi)
    Z = r_monitor * np.cos(Theta)
    
    query_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    interp_x, interp_y, interp_z = interpolators
    
    F_x_sph = interp_x(query_points).reshape(Theta.shape)
    F_y_sph = interp_y(query_points).reshape(Theta.shape)
    F_z_sph = interp_z(query_points).reshape(Theta.shape)
    
    return np.stack((F_x_sph, F_y_sph, F_z_sph), axis=-1)

def compute_vswf_regular(n, m, kr, theta, phi):
    """
    Evaluates the regular Vector Spherical Wave Functions M_nm^(1) and N_nm^(1).
    Returns complex vectors in the Cartesian basis (x, y, z).
    """
    z_n = sp.spherical_jn(n, kr)
    z_n_prime = sp.spherical_jn(n, kr, derivative=True)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    if sin_theta == 0:
        sin_theta = 1e-12 
        
    P_nm = sp.lpmv(m, n, cos_theta)
    
    dP_nm_dtheta = (n * cos_theta * P_nm - (n + m) * sp.lpmv(m, n - 1, cos_theta)) / sin_theta if n > 0 else 0.0
    
    phase = np.exp(1j * m * phi)
    
    M_r = 0.0j
    M_theta = (1j * m * P_nm / sin_theta) * z_n * phase
    M_phi = -dP_nm_dtheta * z_n * phase
    
    N_r = (n * (n + 1) * P_nm * z_n / kr) * phase
    N_theta = dP_nm_dtheta * (z_n + kr * z_n_prime) / kr * phase
    N_phi = (1j * m * P_nm / sin_theta) * (z_n + kr * z_n_prime) / kr * phase
    
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    
    T_sph_to_cart = np.array([
        [sin_theta * cos_p, cos_theta * cos_p, -sin_p],
        [sin_theta * sin_p, cos_theta * sin_p,  cos_p],
        [cos_theta,        -sin_theta,         0.0   ]
    ])
    
    M_cart = T_sph_to_cart @ np.array([M_r, M_theta, M_phi])
    N_cart = T_sph_to_cart @ np.array([N_r, N_theta, N_phi])
    
    return M_cart, N_cart

def compute_incident_coefficients(n, m, theta_inc, phi_inc, pol_vector, P_mn):
    """
    Analytically derives the incident VSWF expansion coefficients (a_nm, b_nm) 
    for an incident plane wave propagating in the direction (theta_inc, phi_inc) 
    with a normalized polarization vector (pol_vector).
    """
    cos_theta = np.cos(theta_inc)
    sin_theta = np.sin(theta_inc)
    
    if sin_theta == 0:
        sin_theta = 1e-12
        
    P_nm = sp.lpmv(m, n, cos_theta)
    dP_nm_dtheta = (n * cos_theta * P_nm - (n + m) * sp.lpmv(m, n - 1, cos_theta)) / sin_theta if n > 0 else 0.0
    
    phase = np.exp(-1j * m * phi_inc)
    
    C_nm_theta = (1j * m * P_nm / sin_theta) * phase
    C_nm_phi = -dP_nm_dtheta * phase
    
    B_nm_theta = dP_nm_dtheta * phase
    B_nm_phi = (1j * m * P_nm / sin_theta) * phase
    
    sin_p, cos_p = np.sin(phi_inc), np.cos(phi_inc)
    
    T_cart_to_sph = np.array([
        [sin_theta * cos_p, sin_theta * sin_p, cos_theta],
        [cos_theta * cos_p, cos_theta * sin_p, -sin_theta],
        [-sin_p,            cos_p,             0.0]
    ])
    
    pol_sph = T_cart_to_sph @ pol_vector
    e_theta = pol_sph[1]
    e_phi = pol_sph[2]
    
    prefactor = 4 * np.pi * ((-1j)**n) / P_mn
    
    a_nm = prefactor * (C_nm_theta * e_theta + C_nm_phi * e_phi)
    b_nm = prefactor * (B_nm_theta * e_theta + B_nm_phi * e_phi)
    
    return a_nm, b_nm

def compute_surface_integral(E_sca, H_sca, n, m, r_monitor, theta_grid, phi_grid, weights_theta, weights_phi, k, P_mn, mode_type, omega, mu):
    """
    Executes the numerical surface integration defined in Eq. (60) utilizing Gauss-Legendre quadrature.
    """
    integral_sum = 0.0j
    kr = k * r_monitor
    
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            n_hat = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            M_nm, N_nm = compute_vswf_regular(n, m, kr, theta, phi)
            M_nm_star = np.conjugate(M_nm)
            N_nm_star = np.conjugate(N_nm)
            
            E_field = E_sca[i, j]
            H_field = H_sca[i, j]
            
            if mode_type == 'M':
                Psi_star = M_nm_star
                curl_Psi_star = k * N_nm_star
            else:
                Psi_star = N_nm_star
                curl_Psi_star = k * M_nm_star
                
            curl_E_sca = -1j * omega * mu * H_field
            
            term1 = np.cross(E_field, curl_Psi_star)
            term2 = np.cross(Psi_star, curl_E_sca)
            
            integrand = np.dot((term1 + term2), n_hat)
            
            differential_area = (r_monitor**2) * weights_theta[i] * weights_phi
            integral_sum += integrand * differential_area
            
    return integral_sum / P_mn

def extract_transition_matrix(F_matrix, A_matrix):
    """
    Extracts the robust transition matrix utilizing the Moore-Penrose pseudoinverse.
    T = [F][A]^\dagger ([A][A]^\dagger)^-1
    """
    A_pinv = np.linalg.pinv(A_matrix)
    T_matrix = F_matrix @ A_pinv
    return T_matrix

def main(l_max, r_monitor, freq, metadata_filepath):
    """
    Ingests dynamic Lebedev nodal data to invert the Foldy-Lax multiple scattering equations.
    
    Parameters:
    - l_max: Maximum polar degree for multipole truncation
    - r_monitor: Radius of the spherical evaluation boundary
    - freq: Operating frequency in Hz
    - metadata_filepath: Path to the JSON manifest generated by the extraction script
    """
    c = 299792458.0
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * freq
    k = omega / c
    
    # Load serialized excitation parameters
    with open(metadata_filepath, 'r') as f:
        excitations = json.load(f)
    
    theta_grid, phi_grid, w_theta, w_phi = generate_gauss_legendre_grid(l_max)
    
    N_dim = 2 * l_max * (l_max + 2)
    num_excitations = len(excitations)
    
    F_matrix = np.zeros((N_dim, num_excitations), dtype=complex)
    A_matrix = np.zeros((N_dim, num_excitations), dtype=complex)
    
    for col_idx, exc in enumerate(excitations):
        
        # Ingest parameters
        e_file = exc['e_file']
        h_file = exc['h_file']
        theta_inc_deg = exc['theta_inc']
        phi_inc_deg = exc['phi_inc']
        pol_type = exc['polarization']
        
        theta_inc = math.radians(theta_inc_deg)
        phi_inc = math.radians(phi_inc_deg)
        
        # Reconstruct orthogonal Cartesian polarization vectors
        if pol_type == 'TE':
            pol_vector = np.array([-math.sin(phi_inc), math.cos(phi_inc), 0.0])
        elif pol_type == 'TM':
            pol_vector = np.array([math.cos(theta_inc)*math.cos(phi_inc), math.cos(theta_inc)*math.sin(phi_inc), -math.sin(theta_inc)])
        else:
            raise ValueError(f"Strict mathematical evaluation failed: Unknown polarization {pol_type}")
        
        # Parse Cartesian voxel data for both fields
        x_ax, y_ax, z_ax, E_cart = parse_cst_ascii_export(e_file)
        _, _, _, H_cart = parse_cst_ascii_export(h_file)
        
        E_interpolators = build_cartesian_interpolators(x_ax, y_ax, z_ax, E_cart)
        H_interpolators = build_cartesian_interpolators(x_ax, y_ax, z_ax, H_cart)
        
        E_sca = interpolate_to_spherical_boundary(E_interpolators, r_monitor, theta_grid, phi_grid)
        H_sca = interpolate_to_spherical_boundary(H_interpolators, r_monitor, theta_grid, phi_grid)
        
        row_idx = 0
        for n in range(1, l_max + 1):
            for m in range(-n, n + 1):
                P_mn = n * (n + 1) * (4 * np.pi) / (2 * n + 1) * math.factorial(n + m) / math.factorial(n - m)
                
                # 1. Populate Scattered Field Matrix [F] via Numerical Integration
                f_mn_M = compute_surface_integral(E_sca, H_sca, n, m, r_monitor, theta_grid, phi_grid, w_theta, w_phi, k, P_mn, 'M', omega, mu_0)
                f_mn_N = compute_surface_integral(E_sca, H_sca, n, m, r_monitor, theta_grid, phi_grid, w_theta, w_phi, k, P_mn, 'N', omega, mu_0)
                
                F_matrix[row_idx, col_idx] = f_mn_M
                F_matrix[row_idx + 1, col_idx] = f_mn_N
                
                # 2. Populate Incident Field Matrix [A] via Analytical Expansion
                a_nm, b_nm = compute_incident_coefficients(n, m, theta_inc, phi_inc, pol_vector, P_mn)
                
                A_matrix[row_idx, col_idx] = a_nm
                A_matrix[row_idx + 1, col_idx] = b_nm
                
                row_idx += 2
                
    T_operator = extract_transition_matrix(F_matrix, A_matrix)
    print("Transition matrix extraction complete.")
    return T_operator

if __name__ == "__main__":
    L_MAX = 5
    R_MONITOR = 10e-3
    FREQUENCY = 10e9
    OUTPUT_DIRECTORY = r"C:\Research\Metamaterials\T_Matrices"
    METADATA_FILE = r"C:\Research\Metamaterials\Exported_Fields\excitation_metadata.json"
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Extract the Transition Matrix
    T = main(L_MAX, R_MONITOR, FREQUENCY, METADATA_FILE)
    
    # Serialize and persist the complex matrix to disk
    output_filepath = os.path.join(OUTPUT_DIRECTORY, "T_Matrix_UnitCell_A.npy")
    np.save(output_filepath, T)
    print(f"Transition matrix successfully serialized to: {output_filepath}")