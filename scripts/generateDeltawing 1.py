# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
assert SUAVE.__version__ == '2.5.2', 'These tutorials only work with the SUAVE 2.5.2 release'
from SUAVE.Core import Units, Data 
import sys
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_segmented_planform
from SUAVE.Plots.Geometry import *
from SUAVE.Input_Output.OpenVSP import write
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
VLM_path = '/home/yiren/Desktop/VortexNet/SUAVE/'
sys.path.append(VLM_path)
from VLM import VLM
import pylab as plt
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os


"""
Code to build Delta wing using SUAVE library 
(c) Yiren Shen

Initial Date: Oct 20, 2024

Modification: 
"""

class NACA4DigitAirfoil:
    def __init__(self, m=0, p=0, t=12, chord_length=1.0):
        """
        Parameters:
        - m: Maximum camber as percentage of chord (e.g., 2 for 0.02 or 2%)
        - p: Position of maximum camber as tenth of chord (e.g., 4 for 0.4 or 40%)
        - t: Maximum thickness as percentage of chord (e.g., 12 for 0.12 or 12%)
        - chord_length: Length of the airfoil chord (default: 1.0)
        """
        self.m = m / 100.0
        self.p = p / 10.0
        self.t = t / 100.0
        self.chord_length = chord_length

    def thickness_distribution(self, x):
        """
        Thickness distribution for the NACA 4-digit airfoil.
        """
        t = self.t
        return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                        0.2843 * x**3 - 0.1036 * x**4)

    def camber_line(self, x):
        """
        Camber line for the NACA 4-digit airfoil.
        """
        m, p = self.m, self.p
        yc = np.where(
            x < p,
            m * (2 * p * x - x**2) / (p**2) if p != 0 else 0,
            m * (1 - 2 * p + 2 * p * x - x**2) / ((1 - p)**2) if p != 0 else 0
        )
        return yc

    def camber_slope(self, x):
        """
        Derivative of the camber line.
        """
        m, p = self.m, self.p
        dyc_dx = np.where(
            x < p,
            2 * m * (p - x) / (p**2) if p != 0 else 0,
            2 * m * (p - x) / ((1 - p)**2) if p != 0 else 0
        )
        return dyc_dx

    def generate_coordinates(self, num_points=100):
        """
        Generate upper and lower surface coordinates for the NACA 4-digit airfoil.
        """
        x = np.linspace(0, 1, num_points)
        yt = self.thickness_distribution(x)
        yc = self.camber_line(x)
        dyc_dx = self.camber_slope(x)

        theta = np.arctan(dyc_dx)

        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine coordinates for airfoil plot
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])

        return x_coords * self.chord_length, y_coords * self.chord_length

    def write_coordinates(self, filename):
        """
        Write airfoil coordinates to a file.
        """
        x, y = self.generate_coordinates()
        with open(filename, 'w') as f:
            f.write(f'NACA{int(self.m * 100)}{int(self.p * 10)}{int(self.t * 100)}\n')
            for xi, yi in zip(x, y):
                f.write(f'{xi:.6f} {yi:.6f}\n')

def vehicle_setup(LE_SWEEP, NACA_4DIGITS, root_chord_in=25.734,
                  twist_root=0.0, twist_tip=0.0, dihedral=0.0):
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'DeltaWing'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # basic parameters
    vehicle.mass_properties.center_of_gravity =  [[0.00001, 0.0, 0.0]] 

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    # wing geom
    wing.sweeps.leading_edge = LE_SWEEP * Units.deg
    wing.taper                   = 0.0
    wing.chords.root             = root_chord_in * Units.inches
    wing.total_length            = root_chord_in * Units.inches
    wing.chords.tip              = 0.0 * Units.meter
    mac_in = (2.0 / 3.0) * root_chord_in
    wing.chords.mean_aerodynamic = mac_in * Units.inches
    wing.origin                  = [[0.0, 0.0, 0.0]] 
    ac_x = (2.0 / 3.0) * root_chord_in * Units.inches
    wing.aerodynamic_center      = [ac_x, 0.0, 0.0]
    semi_span = wing.chords.root / np.tan(LE_SWEEP * Units.deg)
    wing.spans.projected         = semi_span * 2
    wing.areas.reference         = wing.chords.root * semi_span
    wing.aspect_ratio            = (wing.spans.projected)**2 / wing.areas.reference
    wing.thickness_to_chord      = NACA_4DIGITS['t'] / 100.0
    wing.areas.wetted            = 2 * wing.areas.reference
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False
    wing.vortex_lift             = True
    wing.high_mach               = True
    wing.dynamic_pressure_ratio  = 1.0
    # compute quarter chord sweep
    wing_sweeps_quarter_chord = convert_sweep(wing, old_ref_chord_fraction=0.0, new_ref_chord_fraction=0.25)

    # Generate the NACA 4-digit airfoil
    airfoil_generator = NACA4DigitAirfoil(
        m=NACA_4DIGITS['m'],
        p=NACA_4DIGITS['p'],
        t=NACA_4DIGITS['t'],
        chord_length=NACA_4DIGITS['chord_length']
    )

    # Write airfoil coordinates to a file
    airfoil_filename = f"NACA{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}.dat"
    airfoil_generator.write_coordinates(airfoil_filename)

    # Create an airfoil object
    airfoil = SUAVE.Components.Airfoils.Airfoil()
    airfoil.coordinate_file = airfoil_filename

    # Wing Segments
    # Segment 1: Root
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                             = 'Root'
    segment.percent_span_location           = 0.0
    segment.twist                           = twist_root * Units.deg
    segment.root_chord_percent              = 1.0
    segment.dihedral_outboard               = dihedral * Units.deg
    segment.sweeps.quarter_chord            = wing_sweeps_quarter_chord
    segment.thickness_to_chord              = wing.thickness_to_chord
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)

    # Segment 2: Tip
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                           = 'Tip'
    segment.percent_span_location         = 1.0
    segment.twist                         = twist_tip * Units.deg
    segment.root_chord_percent            = 0.0
    segment.dihedral_outboard             = dihedral * Units.degrees
    segment.sweeps.quarter_chord          = wing_sweeps_quarter_chord
    segment.thickness_to_chord            = wing.thickness_to_chord
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)

    # Update wing planform
    wing = wing_segmented_planform(wing)

    # converted from vsp file assuming original unit in inch
    vehicle.reference_area               = wing.areas.reference 
    vehicle.total_length                 = wing.chords.root

    # Add wing to vehicle
    vehicle.append_component(wing)

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ----------------------------------------------------------------------
def full_setup(LE_SWEEP, NACA_4DIGITS):
    PLOT = False
    # vehicle data
    vehicle = vehicle_setup(LE_SWEEP, NACA_4DIGITS)
    if PLOT:
        SUAVE.Plots.Geometry.plot_vehicle(vehicle)
    configs  = configs_setup(vehicle, LE_SWEEP, NACA_4DIGITS)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)


    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses

    
    return configs, analyses

def configs_setup(vehicle, LE_SWEEP, NACA_4DIGITS):
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = f"deltawing_sweep_{LE_SWEEP}_naca_{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}"
    configs.append(base_config)
    #write(vehicle, base_config.tag, write_igs=True)

    return configs


def generate_family_of_delta_wings(le_sweep_angles, naca_variations):
    for le_sweep in le_sweep_angles:
        for naca_params in naca_variations:

            # Define LE_SWEEP and NACA_4DIGITS
            LE_SWEEP = le_sweep
            NACA_4DIGITS = naca_params

            # Setup vehicle with specified sweep angle and airfoil
            vehicle = vehicle_setup(LE_SWEEP, NACA_4DIGITS)
            configs = configs_setup(vehicle, LE_SWEEP, NACA_4DIGITS)

            config_tag = f"deltawing_sweep_{LE_SWEEP}_naca_{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}"
            base_config = configs[config_tag]
            print(f"Generated configuration: {config_tag}")

    return 0

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses
    
# ----------------------------------------------------------------------
#   VLM Settings
# ----------------------------------------------------------------------
def get_settings():
    settings = SUAVE.Analyses.Aerodynamics.Vortex_Lattice().settings
    settings.number_spanwise_vortices        = 15
    settings.number_chordwise_vortices       = 30   
    settings.propeller_wake_model            = None
    settings.spanwise_cosine_spacing         = True
    settings.model_fuselage                  = True
    settings.model_nacelle                   = True
    settings.leading_edge_suction_multiplier = 1
    settings.discretize_control_surfaces     = False
    settings.use_VORLAX_matrix_calculation   = True    
                
    #misc settings
    settings.show_prints = False
    
    return settings

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()    # Aerodynamics.
    #aerodynamics.compute.lift.inviscid_wings.settings.leading_edge_suction_multiplier = -1    #VLM correction for delta wing, check the NACA CR 2865
    aerodynamics.geometry = vehicle    
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.span_efficiency            = .8
    
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    
    # Not yet available for this configuration

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks #what is called throughout the mission (at every time step))
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    # done!
    return analyses    

def get_conditions(machs, altitudes, aoas, PSIs, PITCHQs, YAWQs, ROLLQs):

    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    atmosphere                              = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    speeds_of_sound                         = atmosphere.compute_values(altitudes).speed_of_sound
    v_infs                                  = machs * speeds_of_sound.flatten() 
    conditions.freestream.velocity          = np.atleast_2d(v_infs).T 
    conditions.freestream.mach_number       = np.atleast_2d(machs).T 
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T 
    conditions.aerodynamics.side_slip_angle = np.atleast_2d(PSIs).T 
    conditions.stability.dynamic.pitch_rate = np.atleast_2d(PITCHQs).T 
    conditions.stability.dynamic.roll_rate  = np.atleast_2d(ROLLQs).T 
    conditions.stability.dynamic.yaw_rate   = np.atleast_2d(YAWQs).T 
    
    return conditions

            
def point_analysis(vehicle, AOA, Ma, LE_SWEEP, NACA_4DIGITS, if_plot = False,
                   DCP_overwrite=None, SPC_enforce = -1,
                   root_chord_in=25.734, twist_root=0.0, twist_tip=0.0, dihedral=0.0):
    """
    Function perform analysis for a single point
    Return and examine the panel lift distribution 
    """

    alpha = AOA * Units.deg
    pitch_rate = 0.0 * Units.deg
    yaw_rate = 0.0 * Units.deg
    roll_rate = 0.0 * Units.deg
    M_inf = Ma
    Alt = 5000 * Units.ft
    PSIs = 0.0 * Units.deg

    # get settings and conditions
    conditions = get_conditions(M_inf, Alt, alpha, PSIs, pitch_rate, yaw_rate, roll_rate)
    settings   = get_settings()

    # run VLM
    geometry    = vehicle_setup(LE_SWEEP, NACA_4DIGITS,
                                root_chord_in=root_chord_in, twist_root=twist_root,
                                twist_tip=twist_tip, dihedral=dihedral)
    data        = VLM(conditions, settings, geometry, DCP_overwrite=DCP_overwrite, SPC_enforce = SPC_enforce)
    plot_title  = geometry.tag

    # save/load results
    results = Data()
    results.CL         =  data.CL
    results.CDi        =  data.CDi
    results.CM         =  data.CM
    results.CYTOT      =  data.CYTOT        # Total y force coeff
    results.CRTOT      =  data.CRTOT        # Rolling moment coeff (unscaled)
    results.CRMTOT     =  data.CRMTOT       # Rolling moment coeff (scaled by w_span)
    results.CNTOT      =  data.CNTOT        # Yawing  moment coeff (unscaled)
    results.CYMTOT     =  data.CYMTOT       # Yawing  moment coeff (scaled by w_span)
    results.VD         =  data.VD           # Votext location coordinate
    results.V_distribution = data.V_distribution # Free stream velocity 
    results.gamma       = data.gamma # circulation distribution
    results.cp          = data.CP # pressure coefficient distribution   
    results.alpha_local = data.alpha_local
    results.beta_local  = data.beta_local
    results.gamma_local = data.gamma_local
    results.theta_x     = data.theta_x
    results.theta_y     = data.theta_y
    results.theta_z     = data.theta_z
    results.v_x          = data.v_total_x
    results.v_y          = data.v_total_y
    results.v_z          = data.v_total_z
    results.A = data.A
    results.RHS = data.RHS
    results.RNMAX = data.RNMAX
    results.CHORD = data.CHORD
    results.DCPSID = data.DCPSID
    results.FACTOR = data.FACTOR


    if if_plot:
        # plot results
        plot_vortex_lattice(results.VD, 'Vortex Lattice Locations')
        # plot pressure coefficient 
        plot_field_distribution(results.VD, results.cp, 'CP Distribution')
        # plot circulation distribution
        plot_field_distribution(results.VD, results.gamma, 'Circulation Distribution')

    return results

def main():
    # Define geom and stream conditions
    LE_SWEEP = 55
    NACA_4DIGITS = {'m': 0, 'p': 0, 't': 10, 'chord_length': 1.0}
    AOA = float(10)
    Ma = float(0.3)
    # Config name 
    current_dir = os.getcwd()
    config_name = f"deltawing_sweep_{LE_SWEEP}_naca_{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}"
    print(f"Current Configuration: {config_name}")
    
    
    # SUAVE components 
    configs, analyses = full_setup(LE_SWEEP, NACA_4DIGITS)
    configs.finalize()
    analyses.finalize()
    point_result = point_analysis(configs[config_name], AOA, Ma, LE_SWEEP, NACA_4DIGITS, if_plot = False, DCP_overwrite=None, SPC_enforce = -1)
    vlm_cm = point_result.CM
    vlm_cl = point_result.CL
    vlm_cd = point_result.CDi
    print(f"VLM Simulation for Configuration {config_name}")
    print(f"CL: {vlm_cl}, CD: {vlm_cd}, CM: {vlm_cm}")
    
    
if __name__ == '__main__':
    main()
