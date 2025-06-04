import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

def householder_reflection_3d(G, v):
    """Calculate the Householder reflection of 3D vector G across the hyperplane with normal v."""
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return G
    v_unit = v / v_norm
    v_dot_G = np.dot(v_unit, G)
    return G - 2 * v_dot_G * v_unit

def project_3d_to_2d(G_3d, projection_type='xy'):
    """Project 3D vector to 2D using different projection methods."""
    if projection_type == 'xy':
        return np.array([G_3d[0], G_3d[1]])
    elif projection_type == 'xz':
        return np.array([G_3d[0], G_3d[2]])
    elif projection_type == 'yz':
        return np.array([G_3d[1], G_3d[2]])
    elif projection_type == 'stereographic':
        # Stereographic projection from unit sphere
        if G_3d[2] >= 1.0 - 1e-10:  # Near north pole
            return np.array([0, 0])
        denom = 1 - G_3d[2]
        return np.array([G_3d[0]/denom, G_3d[1]/denom])
    elif projection_type == 'orthographic':
        # Orthographic projection with viewing angle
        return np.array([G_3d[0]*0.8 + G_3d[2]*0.2, G_3d[1]*0.8 + G_3d[2]*0.2])

def stereographic_inverse(S_2d):
    """Recover 3D point from 2D stereographic projection - EXACT inversion"""
    if np.linalg.norm(S_2d) < 1e-10:
        # Point at origin maps to north pole
        return np.array([0, 0, 1])
    
    norm_sq = np.dot(S_2d, S_2d)
    denom = 1 + norm_sq
    
    x = 2 * S_2d[0] / denom
    y = 2 * S_2d[1] / denom
    z = (norm_sq - 1) / denom
    
    return np.array([x, y, z])

def sacred_geometry_embedding(S_2d, embedding_type='golden_sphere'):
    """Embed 2D S back to 3D using sacred geometry principles."""
    if np.linalg.norm(S_2d) < 1e-10:
        return np.array([0, 0, 1])
    
    if embedding_type == 'golden_sphere':
        # Golden ratio height embedding
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        z = 1/phi  # Golden ratio height
        
        # Scale S to fit on sphere at this height
        max_radius_at_z = np.sqrt(1 - z**2)
        s_norm = np.linalg.norm(S_2d)
        if s_norm > 0:
            scale = max_radius_at_z / s_norm
            return np.array([S_2d[0] * scale, S_2d[1] * scale, z])
        else:
            return np.array([0, 0, z])
    
    elif embedding_type == 'unit_sphere':
        # Project onto unit sphere
        s_norm = np.linalg.norm(S_2d)
        if s_norm > 0:
            # Use S magnitude to determine Z coordinate
            z = np.tanh(s_norm - 1)  # Maps [0,∞) to [-1,1) smoothly
            max_radius = np.sqrt(1 - z**2)
            scale = max_radius / s_norm
            return np.array([S_2d[0] * scale, S_2d[1] * scale, z])
        else:
            return np.array([0, 0, 0])
    
    elif embedding_type == 'cone':
        # Cone embedding - S becomes base, height from magnitude
        height = np.tanh(np.linalg.norm(S_2d))  # Bounded height
        return np.array([S_2d[0], S_2d[1], height])

def calculate_psi_3d_2d(G_3d, S_2d, alpha, beta):
    """Calculate Ψ combining 3D G and 2D S."""
    # Project G to 2D for combination
    G_2d = project_3d_to_2d(G_3d)
    psi_2d = alpha * G_2d - beta * S_2d
    return psi_2d

def spherical_to_cartesian(theta, phi, r=1):
    """Convert spherical coordinates to 3D cartesian."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def cartesian_to_spherical(x, y, z):
    """Convert 3D cartesian to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r) if r > 0 else 0
    return theta, phi, r

def check_on_nuit_boundary(point, radius=1.0, tolerance=0.05):
    """Check if a point is on or near the Nuit boundary circle."""
    distance_from_origin = np.linalg.norm(point)
    distance_from_circle = abs(distance_from_origin - radius)
    return distance_from_circle <= tolerance, distance_from_origin, distance_from_circle

def main():
    # Create figure with subplots for 3D and 2D views
    fig = plt.figure(figsize=(20, 10))
    
    # 3D subplot for G vector
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D G Vector Space')
    
    # 2D subplot for S projection and Psi
    ax2 = fig.add_subplot(132)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('2D S Projection Space with Nuit Boundary')
    
    # 3D subplot for S reconstruction
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_zlim(-1.5, 1.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D S Reconstruction')
    
    plt.subplots_adjust(left=0.25, bottom=0.45)
    
    # Initial values
    initial_theta = np.pi/4  # 45 degrees
    initial_phi = np.pi/3    # 60 degrees
    initial_hadit_theta = 0
    initial_hadit_phi = np.pi/2
    initial_alpha = 0.5
    initial_beta = 0.5
    initial_nuit_radius = 1.0
    projection_type = 'stereographic'  # Default to stereographic for reconstruction
    
    # Initial 3D G vector
    G_3d = spherical_to_cartesian(initial_theta, initial_phi)
    
    # Initial 3D Hadit vector
    hadit_3d = spherical_to_cartesian(initial_hadit_theta, initial_hadit_phi)
    
    # Calculate S as 2D projection of reflected G
    G_reflected = householder_reflection_3d(G_3d, hadit_3d)
    S_2d = project_3d_to_2d(G_reflected, projection_type)
    
    # Calculate Psi
    psi_2d = calculate_psi_3d_2d(G_3d, S_2d, initial_alpha, initial_beta)
    
    # Draw initial 3D elements
    def draw_3d_sphere(ax, alpha=0.1):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=alpha, color='gray')
    
    draw_3d_sphere(ax1)
    draw_3d_sphere(ax3)
    
    # Sliders
    slider_height = 0.02
    slider_spacing = 0.03
    slider_width = 0.25
    left_col = 0.05
    mid_col = 0.35
    right_col = 0.65
    
    # Left column sliders
    ax_theta = plt.axes([left_col, 0.35, slider_width, slider_height])
    ax_phi = plt.axes([left_col, 0.35 - slider_spacing, slider_width, slider_height])
    ax_hadit_theta = plt.axes([left_col, 0.35 - 2*slider_spacing, slider_width, slider_height])
    
    # Middle column sliders
    ax_hadit_phi = plt.axes([mid_col, 0.35, slider_width, slider_height])
    ax_alpha = plt.axes([mid_col, 0.35 - slider_spacing, slider_width, slider_height])
    ax_beta = plt.axes([mid_col, 0.35 - 2*slider_spacing, slider_width, slider_height])
    
    # Right column slider
    ax_nuit_radius = plt.axes([right_col, 0.35, slider_width, slider_height])
    
    s_theta = Slider(ax_theta, 'G θ (azimuth)', 0, 2*np.pi, valinit=initial_theta)
    s_phi = Slider(ax_phi, 'G φ (polar)', 0, np.pi, valinit=initial_phi)
    s_hadit_theta = Slider(ax_hadit_theta, 'Hadit θ', 0, 2*np.pi, valinit=initial_hadit_theta)
    s_hadit_phi = Slider(ax_hadit_phi, 'Hadit φ', 0, np.pi, valinit=initial_hadit_phi)
    s_alpha = Slider(ax_alpha, 'α', 0, 2, valinit=initial_alpha)
    s_beta = Slider(ax_beta, 'β', 0, 2, valinit=initial_beta)
    s_nuit_radius = Slider(ax_nuit_radius, 'Nuit Radius', 0.1, 2.0, valinit=initial_nuit_radius)
    
    # Projection type buttons
    ax_proj = plt.axes([0.05, 0.05, 0.15, 0.15])
    proj_button = RadioButtons(
        ax_proj,
        ('XY Plane', 'XZ Plane', 'YZ Plane', 'Stereographic', 'Orthographic'),
        active=3  # Default to Stereographic
    )
    
    # Reconstruction method buttons
    ax_recon = plt.axes([0.25, 0.05, 0.15, 0.15])
    recon_button = RadioButtons(
        ax_recon,
        ('Stereographic Inverse', 'Golden Sphere', 'Unit Sphere', 'Cone'),
        active=0
    )
    
    def update(val=None):
        # Get current values
        theta = s_theta.val
        phi = s_phi.val
        hadit_theta = s_hadit_theta.val
        hadit_phi = s_hadit_phi.val
        alpha = s_alpha.val
        beta = s_beta.val
        nuit_radius = s_nuit_radius.val
        
        # Update 3D vectors
        G_3d = spherical_to_cartesian(theta, phi)
        hadit_3d = spherical_to_cartesian(hadit_theta, hadit_phi)
        G_reflected = householder_reflection_3d(G_3d, hadit_3d)
        
        # Update 2D projections
        proj_map = {'XY Plane': 'xy', 'XZ Plane': 'xz', 'YZ Plane': 'yz', 
                   'Stereographic': 'stereographic', 'Orthographic': 'orthographic'}
        current_proj = proj_map.get(proj_button.value_selected, 'stereographic')
        
        S_2d = project_3d_to_2d(G_reflected, current_proj)
        G_proj_2d = project_3d_to_2d(G_3d, current_proj)
        psi_2d = calculate_psi_3d_2d(G_3d, S_2d, alpha, beta)
        
        # Calculate 3D reconstruction of S
        recon_map = {
            'Stereographic Inverse': 'stereographic',
            'Golden Sphere': 'golden_sphere',
            'Unit Sphere': 'unit_sphere',
            'Cone': 'cone'
        }
        recon_method = recon_map.get(recon_button.value_selected, 'stereographic')
        
        if recon_method == 'stereographic' and current_proj == 'stereographic':
            S_3d_reconstructed = stereographic_inverse(S_2d)
            reconstruction_error = np.linalg.norm(G_reflected - S_3d_reconstructed)
        else:
            S_3d_reconstructed = sacred_geometry_embedding(S_2d, recon_method)
            reconstruction_error = np.linalg.norm(G_reflected - S_3d_reconstructed)
        
        # Check if S is on Nuit boundary
        on_boundary, dist_from_origin, dist_from_circle = check_on_nuit_boundary(S_2d, nuit_radius)
        
        # Update 3D plot (original vectors)
        ax1.clear()
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(-1.5, 1.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        draw_3d_sphere(ax1)
        
        # Draw 3D vectors
        ax1.quiver(0, 0, 0, G_3d[0], G_3d[1], G_3d[2], 
                  color='green', arrow_length_ratio=0.1, linewidth=3, label='G (3D)')
        ax1.quiver(0, 0, 0, hadit_3d[0], hadit_3d[1], hadit_3d[2], 
                  color='black', arrow_length_ratio=0.1, linewidth=2, 
                  linestyle='--', alpha=0.7, label='Hadit (3D)')
        ax1.quiver(0, 0, 0, G_reflected[0], G_reflected[1], G_reflected[2], 
                  color='orange', arrow_length_ratio=0.1, linewidth=2, 
                  alpha=0.7, label='G Reflected')
        
        ax1.legend()
        ax1.set_title(f'3D G Vector Space\nG={G_3d.round(3)}\nG_refl={G_reflected.round(3)}')
        
        # Update 2D plot
        ax2.clear()
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Draw Nuit boundary circle
        nuit_circle = plt.Circle((0, 0), nuit_radius, fill=False, color='blue', 
                                linewidth=2, linestyle='-', label='Nuit Boundary')
        ax2.add_patch(nuit_circle)
        
        # Draw 2D vectors
        ax2.quiver(0, 0, S_2d[0], S_2d[1], angles='xy', scale_units='xy', 
                  scale=1, color='red', label='S (2D Projection)', width=0.005)
        ax2.quiver(0, 0, psi_2d[0], psi_2d[1], angles='xy', scale_units='xy', 
                  scale=1, color='purple', label='Ψ (2D)', width=0.005)
        ax2.quiver(0, 0, G_proj_2d[0], G_proj_2d[1], angles='xy', scale_units='xy', 
                  scale=1, color='lightgreen', alpha=0.5, label='G Projection', width=0.003)
        
        # Mark S endpoint
        if on_boundary:
            ax2.plot(S_2d[0], S_2d[1], 'ro', markersize=12, 
                    markerfacecolor='red', markeredgecolor='darkred', 
                    markeredgewidth=3, label='S (ON Nuit)', zorder=5)
        else:
            ax2.plot(S_2d[0], S_2d[1], 'ro', markersize=10, 
                    markerfacecolor='white', markeredgecolor='red', 
                    markeredgewidth=2, label='S (OFF Nuit)')
        
        boundary_status = "ON" if on_boundary else "OFF"
        status_color = "green" if on_boundary else "red"
        
        ax2.legend()
        ax2.set_title(f'2D S Projection ({current_proj})\n'
                     f'S={S_2d.round(3)}, Ψ={psi_2d.round(3)}\n'
                     f'S is {boundary_status} Nuit boundary', 
                     color=status_color)
        
        # Update 3D reconstruction plot
        ax3.clear()
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_zlim(-1.5, 1.5)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        draw_3d_sphere(ax3)
        
        # Draw original G_reflected for comparison
        ax3.quiver(0, 0, 0, G_reflected[0], G_reflected[1], G_reflected[2], 
                  color='orange', arrow_length_ratio=0.1, linewidth=2, 
                  alpha=0.5, label='G Reflected (original)', linestyle='--')
        
        # Draw reconstructed S
        ax3.quiver(0, 0, 0, S_3d_reconstructed[0], S_3d_reconstructed[1], S_3d_reconstructed[2], 
                  color='red', arrow_length_ratio=0.1, linewidth=3, 
                  label=f'S Reconstructed ({recon_method})')
        
        # Highlight if S is on Nuit boundary
        if on_boundary:
            ax3.scatter([S_3d_reconstructed[0]], [S_3d_reconstructed[1]], [S_3d_reconstructed[2]], 
                       color='gold', s=100, marker='*', label='Nuit Alignment!', zorder=5)
        
        ax3.legend()
        title_color = 'green' if on_boundary else 'black'
        reconstruction_quality = "PERFECT" if reconstruction_error < 1e-10 else f"Error: {reconstruction_error:.6f}"
        
        ax3.set_title(f'3D S Reconstruction ({recon_button.value_selected})\n'
                     f'S_3D={S_3d_reconstructed.round(3)}\n'
                     f'{reconstruction_quality}', color=title_color)
        
        # Add comprehensive info box
        info_text = (f'|S_2D| = {dist_from_origin:.3f}\n'
                    f'Nuit radius = {nuit_radius:.3f}\n'
                    f'Boundary distance = {dist_from_circle:.6f}\n'
                    f'Reconstruction error = {reconstruction_error:.6f}\n'
                    f'Projection: {current_proj}\n'
                    f'Reconstruction: {recon_method}')
        
        if on_boundary:
            info_text += '\n*** S ALIGNS WITH NUIT! ***'
        
        ax2.text(0.02, 0.98, info_text,
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if on_boundary else 'wheat', alpha=0.8))
        
        fig.canvas.draw_idle()
    
    def change_projection(label):
        update()
    
    def change_reconstruction(label):
        update()
    
    # Connect events
    s_theta.on_changed(update)
    s_phi.on_changed(update)
    s_hadit_theta.on_changed(update)
    s_hadit_phi.on_changed(update)
    s_alpha.on_changed(update)
    s_beta.on_changed(update)
    s_nuit_radius.on_changed(update)
    proj_button.on_clicked(change_projection)
    recon_button.on_clicked(change_reconstruction)
    
    # Initial update
    update()
    
    plt.show()

if __name__ == "__main__":
    main()
