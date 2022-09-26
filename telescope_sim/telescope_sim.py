"""
Helper class for building and running multi-aperture telescope simulations with MultiAperturePSFSampler

Author: Ian Cunnyngham (Institute for Astronomy, University of Hawai'i) 2021

Description
- - - - - -

    This package wraps MultiAperturePSFSampler, providing reasonable defaults, building telescope configs,
holding simulation variables like atmosphere, current image (if any), current photon flux, etc. as well
as providing helper functions for running the simulation as well as rendering pretty plots.

This code can be run via any of the following methods:
- Calling this script directly from the CLI, mostly as a way to demo and unit test configurations (see examples below)
- Adding all simulator kwargs to another CLI (see run_dasie_via_gym.py) and then passing the kwargs to the sim builder
- Importing SimulateMultiApertureTelescope class and passing kwargs directly.  (See "equivilant" comments bellow)
  - __init__ kwargs match argparse exactly.  Any missing keywords get their defaults from the argoparse definitions.


Example CLI commands:
- - - - - - - - - - - 

# Default ELF config, accumlate piston, tip, tilt errors each step (Try with --mirror_layout monolithic or keck)
# - Equivilant to telescope_sim = SimulateMultiApertureTelescope()
python simulate_multi_aperture.py --add_ptt_perturbations_sigma .05

# Monolithic 3.6m primary with spiders in a random orientation
# - SimulateMultiApertureTelescope(mirror_layout="monolithic", telescope_radius=1.8, spider_width=.03175)
python simulate_multi_aperture.py --add_ptt_perturbations_sigma .05 --mirror_layout "monolithic" --telescope_radius 1.8 --spider_width .03175

# ELF, default atmosphere, no correction (strehls < .02 typically)
# - SimulateMultiApertureTelescope(atmosphere_type="multi")
python simulate_multi_aperture.py --atmosphere_type "multi"

# Apply instantaneous, best PTT corrections for atmosphere (strehls roughly .4-.7 range)
python simulate_multi_aperture.py --atmosphere_type "multi" --apply_optimal_actuator_corrections

# Same as above, but use DM to simulate PTT (bench demo) (Similar range as above, but slightly less)
# - SimulateMultiApertureTelescope(atmosphere_type="multi", dm_actuator_num=35) 
python simulate_multi_aperture.py --atmosphere_type "multi" --apply_optimal_actuator_corrections --dm_actuator_num=35

# Use direct DM actuation to correct atmosphere as well as best case scenario for AO (Strehls > .9)
# - SimulateMultiApertureTelescope(atmosphere_type="multi", dm_actuator_num=35, directly_actuate_dm=True) 
python simulate_multi_aperture.py --atmosphere_type "multi" --apply_optimal_actuator_corrections --dm_actuator_num=35 --directly_actuate_dm

# Extended object with uncorrected atmosphere, change focal-plane extent to 6 arcsec
# - SimulateMultiApertureTelescope(atmosphere_type="multi", extended_object_image_file="sample_image.png")
python simulate_multi_aperture.py --atmosphere_type "multi" --extended_object_image_file sample_image.png --filter_psf_extent 6

# Load complete MultiAperturePSF setup pkl
python simulate_multi_aperture.py --telescope_setup_pkl=bench_demo.pkl --num_steps=20 --atmosphere_type "multi" --extended_object_image_file=sample_image.png

# Simulate noise by setting the brightness of the object in photons/m^2 (per observation)
python simulate_multi_aperture.py --telescope_setup_pkl=bench_demo.pkl --num_steps=20 --atmosphere_type "multi" --extended_object_image_file=sample_image.png --integrated_photon_flux=1e5

"""


import argparse
import numpy as np
import joblib  # Good .pkl handling
from matplotlib import pyplot as plt

# Load HCIPy and MultiAperturePSFSampler built on top of it
import hcipy 
from .multi_aperture_psf import MultiAperturePSFSampler


### Below is my attempt at not having to redefine and describe every variable many times in the codebase (~Ian C)

def add_multi_aperture_telescope_args(parser):
    """
    Given an argparse.ArgumentParser() instance, append all the args associated with building a multi-aperture simulation
    
    2021-04
    - - - -
    - Added the ability to make different mirror configurations
      - 'monolithic' is a single mirror tha derives it's diamater from telescope_radius
        - Added --central_obscuration_ratio to set how much of the aperture the second blocks (default .25)
      - 'elf' is an annulus of mirrors with their centers at telescope_radius and their subaperture_radius 
              either directly spevified or derived from maximum packing
      - 'keck' A hardcoded pseudo-Keck like setup
    - Added "--directly_actuate_dm" which defaults to False, because there's now the ability to directly send
      DM actuations instead of just using it to approximate PTT actuation.  Can't actuate PTT and DM simultaneously.

    """
    
    ### Extended object setup ###
    # Must match focal-plane resolution if noise is provided
    parser.add_argument('--extended_object_image_file', type=str,
                        help='Filename of image to convolve PSF with (if none, PSF returned)')
    
    
    ### Telescope / pupil-plane setup ###
    
    # For now, passing in telescope setup pkl overrides most everything about telescope config
    parser.add_argument('--telescope_setup_pkl', type=str,
                        help='.pkl file containing dict passed into MultiAperturePSFSampler (overrides CLI telescope arguments)')
    
    ### Primary aperture setup ###
    parser.add_argument('--mirror_layout', type=str,
                        default='elf',
                        help='Telescope layout, can be "elf", "monolithic", "keck", "custom" (Default: "elf")')
    
    parser.add_argument('--mirror_centers', 
                        default=None,
                        help='If layout custom, a pkl file with (2, n_mirror) matrix of x, y mirror center positions (array can also be passed when calling from python)')
    
    parser.add_argument('--num_apertures', type=int,
                        default=15,
                        help='Number of apertures in ELF annulus')

    parser.add_argument('--telescope_radius', type=float,
                        default=1.25,
                        help='Distance from telescope center to aperture centers (meters)')
    
    parser.add_argument('--central_obscuration_ratio', type=float,
                        default=.25,
                        help='(monolithic only) Ratio of the diameter of the central obscuration compared to the pupil diameter')

    parser.add_argument('--subaperture_radius', type=float,
                        default=None,
                        help='Radius of each sub-aperture (default is maximal filling) (meters)')

    parser.add_argument('--spider_width', type=float,
                        default=None,
                        help='Width of spider (default is no spider) (meters)')

    parser.add_argument('--spider_angle', type=float,
                        default=None,
                        help='Spider orientation angle (0-90) (default is random) (degrees)')

    ### Pupil plane ###
    parser.add_argument('--pupil_plane_resolution', type=int,
                        default=2 ** 8,
                        help='Resolution of pupil plane simulation')

    parser.add_argument('--piston_actuate_scale', type=float,
                        default=1e-6,
                        help='Sub-aperture piston actuation scale (meters)')

    parser.add_argument('--tip_tilt_actuate_scale', type=float,
                        default=1e-6,
                        help='Sub-aperture tip and tilt actuation scale (microns/meter)~=(radians)')
    
    
    ### Focal-plane setup ###
    parser.add_argument('--filter_central_wavelength', type=float,
                        default=1e-6,
                        help='Central wavelength of focal-plane observation (meters)')
    
    parser.add_argument('--filter_psf_extent', type=float,
                        default=4.0,
                        help='Angular extent of simulated PSF (arcsec)')
    
    parser.add_argument('--filter_psf_resolution', type=int,
                        default=2 ** 8,
                        help='Resolution of simulated PSF (this and extent set pixel scale for extended image convolution)')
    
    parser.add_argument('--filter_fractional_bandwidth', type=float,
                        default=0.05,
                        help='Fractional bandwidth of filter')
    
    parser.add_argument('--filter_bandwidth_samples', type=int,
                        default=3,
                        help='Number of pupil-planes used to simulate bandwidth (1 = monochromatic)')

    
    ### Atmosphere setup ###
    parser.add_argument('--atmosphere_type', type=str,
                        default="none",
                        help='Atmosphere type: "none" (default), "single" layer, or "multi" layer')
    
    parser.add_argument('--atmosphere_fried_paramater', type=float,
                        default=0.20,
                        help='Fried paramater, r0 @ 550nm (maters)')
    
    parser.add_argument('--atmosphere_outer_scale', type=float,
                        default=200.0,
                        help='Atmosphere outer-scale (maters)')
    
    # !!! Note: Doesn't currentoly work with multi-layer atmospheres, stuck at 10m/s
    parser.add_argument('--atmosphere_velocity', type=float,
                        default=10.0,
                        help='Atmosphere velocity (maters/second)')
    
    # !!! Breaks render right now, but should work for simulation...
    parser.add_argument('--enable_atmosphere_scintillation', action='store_true',
                        default=False,
                        help='Simulate atmospheric scintillation in multi-layer atmosphere')
    
    # Simulate slew speed by imposing extra velocities on each wind layer of above atmosphere
    parser.add_argument('--slew_deg_per_sec', type=float,
                        help='Telescope slew velocity (degrees/second)')
    
    parser.add_argument('--slew_focal_plane_angle', type=float,
                        default=0,
                        help='Direction (w/r/t the focal plane) of slewing (degrees 0-360): Default 0')
    
    ### Object flux and detector noise ###
    
    # In order to get photon noise (and have read noise make sense)
    # we need to specify photon flux integrated over the length of our exposures
    # (photons/m^2).  
    # This can map onto observable magnitudes latter with less assumptions up front
    parser.add_argument('--integrated_photon_flux', type=float,
                        help='Total number of photons/m^2 from FOV (Default: None = no noise)')
    
    # This dpeneds on integrated_photon_flux being specified
    # Not sure that a reasonable default for this is, but there should be *some*
    parser.add_argument('--read_noise', type=float,
                        default=10.0,
                        help='Scaler giving the rms read noise (counts) (Only used when integrated_photon_flux specified)')
    
    
    ### Deformable mirror approximation of PTT actuation ###
    parser.add_argument('--dm_actuator_num', type=int,
                        help='Number of DM actuators on a side (Default: None = no DM)')
    
    parser.add_argument('--dm_actuator_spacing', type=float,
                        default=0.1125,
                        help='pupil-plane spacing of actuators in meters')
    
    parser.add_argument('--directly_actuate_dm',
                        action='store_true',
                        help='If DM is specified above, it can either be used to approximate PTT or actuated directly when sampling (but no simultaneous PTT + DM actuation for now)')
    
    return parser
    

def get_defaults_from_parser(kwargs):
    """Takes a kwargs dictionary, puts it into a format argparser expects, returns the dict with arguments"""

    # Build the parser
    parser = argparse.ArgumentParser()
    parser = add_multi_aperture_telescope_args(parser)
    
    # Parse an empty list
    args = parser.parse_args([])
    
    # For each argument, keep the original key if it exists, else replace it with the default form definitions above
    for key in vars(args):
        kwargs[key] = kwargs.get(key, parser.get_default(key))

    return kwargs


class SimulateMultiApertureTelescope():
    def __init__(self, **kwargs):
        
        # This is how I handle needing default values for kwargs.  I recognize it is a bit of an abuse.
        kwargs = get_defaults_from_parser(kwargs)
        
        # Load a complete telescope configuration dict if provided (Overides CLI args for these params)
        # Or build from CLI args
        self.telescope_setup_pkl = kwargs['telescope_setup_pkl']
        if self.telescope_setup_pkl is not None:
            self.sampler_setup = joblib.load(self.telescope_setup_pkl)
            
            self.num_apertures = self.sampler_setup['mirror_config']['positions'].x.shape[0]
            self.aperture_diamater = self.sampler_setup['mirror_config']['aperture_config'][1]
            self.pupil_plane_diamater = self.sampler_setup['mirror_config']['pupil_extent']
            # CTRL+F'd all the other self. declarations below, and they aren't used anywhere... 
            # Letting it lie for now...
            
        else:
            self.mirror_layout = kwargs['mirror_layout']
            self.telescope_radius = kwargs['telescope_radius']
            
            ### Depending on layout specified, build up primary mirror geometry
            
            if self.mirror_layout == 'monolithic':
                ### Build a single monolithic primary
                
                self.num_apertures = 1
                aperture_diamater = 2 * self.telescope_radius

                # One mirror at center of pupil plane
                self.aperture_centers = hcipy.CartesianGrid(np.array([[0], [0]]))

                # Pupil-plane extent should be a bit larger than the mirror
                self.pupil_plane_diamater = 1.05 * aperture_diamater

                # Specify the aperture config
                self.aperture_config = ['circular_central_obstruction', aperture_diamater, kwargs['central_obscuration_ratio']]

            if self.mirror_layout == 'elf':
                ### Generate ELF-like telescope geometry annulus of sub-apertures with centers at telescope_radius
                self.num_apertures = kwargs['num_apertures']

                # Linear space of angular coordinates for mirror centers
                thetas = np.linspace(0, 2*np.pi, self.num_apertures+1)[:-1]

                # Use HCIPy coordinate generation to quickly generate mirror centers
                aper_coords = hcipy.SeparatedCoords((np.array([self.telescope_radius]), thetas))
                self.ap = aper_coords
                print(aper_coords)

                # Create an HCIPy "CartesianGrid" by creating PolarGrid and converting
                self.aperture_centers = m_cens = hcipy.PolarGrid(aper_coords).as_('cartesian')

                # Calculate subaperure diamater
                self.subaperture_radius = kwargs['subaperture_radius']
                if self.subaperture_radius is not None:
                    aperture_diamater = 2*self.subaperture_radius
                else:
                    # Calculate sub-aperture diamater from the distance between centers 
                    # (Assuming dense packing of apertures for now, could simulate gaps later)
                    aperture_diamater = np.sqrt((m_cens.x[1]-m_cens.x[0])**2 + (m_cens.y[1]-m_cens.y[0])**2)

                # Calculate extent of pupil-plane simulation (meters)
                self.pupil_plane_diamater = max(m_cens.x.max() - m_cens.x.min(), m_cens.y.max() - m_cens.y.min()) + aperture_diamater
                # Add a little extra for edges, not convinced not cutting them off
                self.pupil_plane_diamater *= 1.05  

                self.aperture_config = ['circular', aperture_diamater]
                
            if self.mirror_layout == 'custom_mir_cens':
                ### Generate ELF-like telescope geometry annulus of sub-apertures with centers at telescope_radius
                self.num_apertures = kwargs['num_apertures']

                # Linear space of angular coordinates for mirror centers
                thetas = np.linspace(0, 2*np.pi, self.num_apertures+1)[:-1]

                # Use HCIPy coordinate generation to quickly generate mirror centers
                aper_coords = hcipy.SeparatedCoords((np.array([self.telescope_radius]), thetas))

                # Create an HCIPy "CartesianGrid" by creating PolarGrid and converting
                self.aperture_centers = m_cens = hcipy.PolarGrid(aper_coords).as_('cartesian')

                # Calculate subaperure diamater
                self.subaperture_radius = kwargs['subaperture_radius']
                if self.subaperture_radius is not None:
                    aperture_diamater = 2*self.subaperture_radius
                else:
                    # Calculate sub-aperture diamater from the distance between centers 
                    # (Assuming dense packing of apertures for now, could simulate gaps later)
                    aperture_diamater = np.sqrt((m_cens.x[1]-m_cens.x[0])**2 + (m_cens.y[1]-m_cens.y[0])**2)

                # Calculate extent of pupil-plane simulation (meters)
                self.pupil_plane_diamater = max(m_cens.x.max() - m_cens.x.min(), m_cens.y.max() - m_cens.y.min()) + aperture_diamater
                # Add a little extra for edges, not convinced not cutting them off
                self.pupil_plane_diamater *= 1.05  

                self.aperture_config = ['circular', aperture_diamater]
                
            if self.mirror_layout == 'keck':
                ###  Create a Keck-like layout.  Hardocded for now.
                self.num_apertures = 36

                self.telescope_radius = 10 # meters
                aper_coords = hcipy.make_hexagonal_grid(1.6, 3, False)  # 1.6m between points, 3 rows, pack
                self.aperture_centers = hcipy.CartesianGrid(aper_coords[1:].T) # Remove center coordinate, reform
                self.aperture_config = ['hexagonal', 1.8, np.pi/2]
                self.pupil_plane_diamater = 12
            
            
            ### Configure spider if set
            self.spider_width = kwargs['spider_width']
            self.spider_angle = kwargs['spider_angle']
            if self.spider_width is not None:
                self.spider_config = {
                    'width': self.spider_width,
                }
                # If spider angle is not defined, set it randomly
                if self.spider_angle is None:
                    self.spider_config['random_angle'] = True
                else:
                    self.spider_config['angle'] = self.spider_angle
            else:
                #  If spider_width is None, pass an empty config (no spider)
                self.spider_config = None


            ### Build up non-optional sampler configuration dictionary from kwargs
            
            self.sampler_setup = {
                'mirror_config': {
                    'positions': self.aperture_centers,
                    'aperture_config': self.aperture_config,
                    'pupil_extent': self.pupil_plane_diamater ,
                    'spider_config': self.spider_config,
                    'pupil_res': kwargs['pupil_plane_resolution'],
                    'piston_scale': kwargs['piston_actuate_scale'],   # meters
                    'tip_tilt_scale': kwargs['tip_tilt_actuate_scale']  # meters
                },
                # Single filter for now, sampler designed for with multiple in mind
                'filter_configs': [ 
                {
                    'central_lam': kwargs['filter_central_wavelength'],    # meters
                    'focal_extent': kwargs['filter_psf_extent'],      # arcsec
                    'focal_res': kwargs['filter_psf_resolution'],
                    'frac_bandwidth': kwargs['filter_fractional_bandwidth'],
                    'num_samples': kwargs['filter_bandwidth_samples']
                } ] 
            }

        
        ### If integrated photon flux is set, make sure a detector is setup
        self.int_phot_flux = kwargs['integrated_photon_flux']
        if self.int_phot_flux is not None:
            for i_f, filter_config in enumerate(self.sampler_setup['filter_configs']):
                if 'detector_config' not in filter_config:
                    detector_config = {
                        'read_noise': kwargs['read_noise'],
                        'include_photon_noise': True,
                    }
                    self.sampler_setup['filter_configs'][i_f]['detector_config'] = detector_config
               
        # IF DM config is set, setup for DM approximation of Piston, tip, tilt actuation
        # This is slow, but might be important for fine-tuning the model for bench demo
        self.dm_actuator_num  = kwargs['dm_actuator_num']
        self.direct_dm_actuation = False
        if self.dm_actuator_num is not None:
            self.dm_actuator_spacing  = kwargs['dm_actuator_spacing']
            # arguments passed directly into HCIPY DM construtor
            # [ num_actuators, actuator_pupil_plane_spacing]
            dm_config=[self.dm_actuator_num, self.dm_actuator_spacing]
            self.sampler_setup['mirror_config']['dm_config'] = dm_config
            directly_actuate_dm = kwargs['directly_actuate_dm']
            aprox_ptt_wih_dm = (not directly_actuate_dm)
            self.sampler_setup['mirror_config']['aprox_ptt_wih_dm'] = aprox_ptt_wih_dm
            
            # Make an easily accessible variable to show when the sampler is expecting direct DM actuation
            # For now, simulator.get_observation(..., dm_actuate=...) how you use this.
            if directly_actuate_dm:
                self.direct_dm_actuation = True
            
            
        #print(self.sampler_setup)
            
        ### Initialize sampler with above setup
        self.mas_psf_sampler = MultiAperturePSFSampler(**self.sampler_setup)
        
        
        ### Load extended object if specified
        self.extended_object_image_file = kwargs['extended_object_image_file']
        if self.extended_object_image_file is not None:
            self.set_extended_object( plt.imread( self.extended_object_image_file ) )
        else:
            self.set_extended_object( None )
            

        ### If enabled, setup atmosphere
        self.atmosphere_type = kwargs['atmosphere_type']
        if self.atmosphere_type != "single" and self.atmosphere_type != "multi":
            self.atmos = None
        else:
            self.generate_atmosphere(
                r0 = kwargs["atmosphere_fried_paramater"],
                outer_scale = kwargs["atmosphere_outer_scale"],
                velocity = kwargs["atmosphere_velocity"],
                multi = ( self.atmosphere_type == "multi"),
                scintillation = kwargs['enable_atmosphere_scintillation'],
            )
        
        
        ### If set, simulate slewing by imposing wind-velocities on atmospheres
        self.slew_deg_per_sec = kwargs['slew_deg_per_sec']
        if self.slew_deg_per_sec is not None:
            self.slew_focal_plane_angle = kwargs['slew_focal_plane_angle']
            slew_th = np.pi*self.slew_focal_plane_angle/180
            slew_deg_sec = [ self.slew_deg_per_sec*np.cos(slew_th), self.slew_deg_per_sec*np.sin(slew_th)]
            self.set_atmos_slew_wind(slew_deg_sec)
        
        ### Setup variables for plotting
        
        # Focal plane extent in arcseconds
        f_ext = self.sampler_setup['filter_configs'][0]['focal_extent']
        self.plt_focal_extent = [-.5*f_ext, .5*f_ext, -.5*f_ext, .5*f_ext]
        
        # Find reasonable minimum value for focal plane (minimum of perfectly phased)
        x, _ = self.mas_psf_sampler.sample()
        self.plt_focal_logmin = np.log10(x.min())
        
            
    def set_extended_object(self, image):
        """ Set extended object image to be convolved with telescope PSF """
        # Notes: 
        # - Plate scale (angular pixel size) fixed in simulator
        #   by focal-plane extent and resolution, scale the images?
        # - In current code, if render is enabled, pupil-plane res and image
        #   resolution must match.  This is of course extremely arbitrary
        #   as they absolutely don't need to be related. 
        # - (New: 2020-12): Image and focal plane res need to match when adding
        #                   noise for now.  I should probably fix this later 
        #                   (though it does help keep things explicit)
        self.ext_im = image

        
    def generate_atmosphere(self,
        r0,                  # (meters) Fried paramater: atmosphere coherence length 
        outer_scale = 200,   # (meters) outer scale
        velocity = 10,       # (meters / second) Layer velocity
        multi = False,       # (bool) Whether to use a single turbulence layer or HCIPys multi-layer
        scintillation = False, # (bool) Whether to simulate scintilation
    ):
        """Helper to create an HCIPy atmosphere that will be applied when running sim"""
        
        self.atmosphere_fried_paramater = r0
        self.atmosphere_outer_scale = outer_scale
        self.atmosphere_velocity = velocity
            
        # Calculate C_n^2 from given Fried param, r0 @ 550nm 
        self.cn2 = hcipy.Cn_squared_from_fried_parameter(self.atmosphere_fried_paramater, 550e-9)
            
        if multi:
            # Multi-layer atmosphere
            layers = hcipy.make_standard_atmospheric_layers(
                self.mas_psf_sampler.pupil_grid, 
                self.atmosphere_outer_scale
            )
            for i_l in range(len(layers)):
                # Set velocity of each layer to vector of specified magnitude with random direction
                layers[i_l].velocity = self.mas_psf_sampler._from_mag_gen_rand_vec(self.atmosphere_velocity)
                
            self.atmos = hcipy.MultiLayerAtmosphere(
                layers, 
                scintillation=scintillation
            )
            self.atmos.Cn_squared = self.cn2
            
            self.atmos.reset()
        else:
            # Single layer atmosphere
            self.atmos = hcipy.InfiniteAtmosphericLayer(
                self.mas_psf_sampler.pupil_grid, 
                self.cn2, 
                self.atmosphere_outer_scale, 
                self.atmosphere_velocity, 
                100  # Height of single layer in meters, but may not be important for now
            )
        self.simulation_time = 0
            
    def set_atmos_slew_wind(self, 
        slew_deg_sec  # Two element vector (x, y): slew degrees per second 
    ):
        """Takes a slew vector in deg/sec and imposes pseudo 'wind velocity' to all atmos layers """
    
        if self.atmos is None:
            print("Warning: failed to impose slew speed as no atmosphere is set")
        else:
            # Convert degrees per second into radians
            slew_rad_sec = np.pi*np.array(slew_deg_sec)/180

            # Iterate the old fashioned way since we're modifying the elements
            for i_l in range(len(self.atmos.layers)):
                l = self.atmos.layers[i_l]
                v, h = l.velocity, l.height
                self.atmos.layers[i_l].velocity += h*slew_rad_sec


    def evolve_to(self, 
        simulation_time  # (seconds) Absolute time (from 0) of simulation
    ):
        """Evolve simulation (practically speaking, the atmosphere) until the time specified"""
        
        self.simulation_time = simulation_time
        if self.atmos != None:
            self.atmos.evolve_until( self.simulation_time )
    
    def reset(self):
        """Reset simulation (practically speaking, just the atmosphere)"""
        self.simulation_time = 0
        if self.atmos != None:
            self.atmos.reset()
    
    def get_observation(self,
        piston_tip_tilt = None,  
        dm_actuate = None,
        int_phot_flux = None
    ):
        """
        Return an observation from the telescope simulator given the current state (atmosphere, ext. image if any, photon flux if any)
        
        Inputs
        ------
          piston_tip_tilt : (float) (n_apertures, 3): Piston, tip, and tilt actuation for each sub-aperture as multiplied by the 
                                                      corresponding scales setup during initialization
          dm_actuate : (float) (n_active_actuators) : If DM is setup and direct actuation enable, accepts piston actuation for all active actuators
          int_phot_flux : (float) : (optional) Set a new photon flux for this observation (photons/m^2)
          
        Outputs
        -------
        
        X: Stack of focal plane observations.  PSF by default, extended image convolved with PSF if provided
        Y: Returns the optimal P/T/T (n_aper, 3) phases to get optimal strehl (measured vs atmosphere)
        strehls: If meas_strehls set, returns strehl vs perfectly phase mirror
        
        """
        if int_phot_flux is not None:
            self.int_phot_flux = int_phot_flux
        

        X, Y, strehls  = self.mas_psf_sampler.sample(
            piston_tip_tilt,  # (n_aper, 3) piston, tip, tilts to set telescope to
            dm_actuate=dm_actuate,
            atmos=self.atmos,    # Pass in HCIPy atmosphere, applied to each pupil-plane (or None is fine)
            convolve_im=self.ext_im, # Image to convolve PSF with 
                                     # (Note: assuemd matches sampler filters angular extent/pixel scale)
            int_phot_flux=self.int_phot_flux,  # Photons/m^2 for the entire FOV
            meas_strehl=True     # If True, returns third output which is the measured strehl for each filter 
        )
        
        return X, Y, strehls
    
    def get_integrated_frame(self,
        integration_time = 1,         # Integration time in seconds
        n_subframes = 20,             # number of subframes to build up frame
        piston_tip_tilt = None,       # Fixed actuation to apply
        dm_actuate = None,
        int_phot_flux = None,
        render_subframes=False
    ):
        if int_phot_flux is not None:
            self.int_phot_flux = int_phot_flux
        
        t0 = self.simulation_time
        
        sub_times = np.linspace(t0, t0+integration_time, n_subframes+1)[1:]
        
        all_Ys, all_strehls = [], []
        for i_t, t_sub in enumerate(sub_times):
            
            self.evolve_to(t_sub)
            
            X, Y, strehls  = self.mas_psf_sampler.sample(
                piston_tip_tilt,  # (n_aper, 3) piston, tip, tilts to set telescope to
                dm_actuate=dm_actuate,
                atmos=self.atmos,    # Pass in HCIPy atmosphere, applied to each pupil-plane (or None is fine)
                convolve_im=self.ext_im, # Image to convolve PSF with 
                                         # (Note: assuemd matches sampler filters angular extent/pixel scale)
                int_phot_flux=None,
                meas_strehl=True     # If True, returns third output which is the measured strehl for each filter 
            )
            all_Ys += [ Y ]
            all_strehls += [ strehls ]
            
            if render_subframes:
                plt.figure(figsize=[12, 4])
                self.render(X, strehls)
                plt.show()
            
            if i_t == 0:
                int_frame = np.copy(X)
            else:
                int_frame += X
                
        noise_Xs = []
        for i_samp in range(X.shape[2]):
            noise_samp = self.mas_psf_sampler._addNoiseToObservation(
                observation=int_frame[..., i_samp], 
                int_phot_flux=self.int_phot_flux
            )
            noise_Xs += [ noise_samp[..., None] ]
        final_obs = np.concatenate(noise_Xs, axis=2)
        
        self.simulation_time += integration_time
        
        return final_obs, all_Ys, all_strehls
        
    
    def pupil_plane_phase_screen(self, np_array=False):
        """Returns the pupil-plane phase screen"""
        return self.mas_psf_sampler.getPhaseScreen(self.atmos, np_array=np_array)
        
    def render(self, X, strehls=None):
        """Plots the current observation"""
        
        plt.clf()

        ### Getting the phase screens to plot isn't as pretty as I'd like
        awf1 = self.pupil_plane_phase_screen()
        
        plt.subplot(121)
        hcipy.imshow_field(awf1, mask=self.mas_psf_sampler.aper, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
        plt.ylabel('pupil plane (m)')
        plt.colorbar()

        
        ### Plot pupil and PSF with this atmosphere
        obs = X[..., 0]
        
        plt.subplot(122)
        im = plt.imshow(np.log10(obs), vmin=self.plt_focal_logmin, cmap='inferno', extent=self.plt_focal_extent)
        plt.ylabel('focal plane (arcsec)')
        cbar = plt.colorbar(im)
        
        if strehls is not None:
            plt.title(f'strehl {strehls[0]:.03f}')
        
        plt.pause(0.1)
        


def cli_main(flags):
    
    # Turning namespace key, values into dict
    kwargs = vars(flags)
    
    # Pull out keywards only relevant to this runtime
    n_steps = kwargs.pop('num_steps')
    t_delta = kwargs.pop('step_time_granularity')
    add_ptt_errs_sigma = kwargs.pop('add_ptt_perturbations_sigma')
    apply_corrections = kwargs.pop("apply_optimal_actuator_corrections")
    render = (not kwargs.pop('no_render'))
    
    # Build simulator from the rest of the keyword arguments
    telescope_sim = SimulateMultiApertureTelescope(**kwargs)
    
    # Shape of piston, tip, tilt actuation
    ptt_shape = (telescope_sim.num_apertures, 3)
    ptt_actuation = np.zeros(ptt_shape)
    
    # Setup plot if rendering is enabled
    if render:
        plt.figure(figsize=[12,4])
        plt.show(block=False)
    
    # Iterate through all time-steps
    ts = np.linspace(0, n_steps*t_delta, n_steps, endpoint=False)
    for t in ts:
        print(t, end=' ')
        
        telescope_sim.evolve_to(t)
        
        # Add random errors (shouldn't do anything if 0 (default))
        if add_ptt_errs_sigma is not None:
            ptt_actuation += np.random.normal(0, add_ptt_errs_sigma, ptt_shape)
        
        X, Y, strehls = telescope_sim.get_observation(ptt_actuation)
        
        # If --apply_optimal_actuator_corrections is set, run the simulation again with optimal actuator fits
        if apply_corrections:
            if telescope_sim.direct_dm_actuation:
                # If direct DM actuation is setup, Y will be the optimal fit of the pupil plane errors
                X, Y, strehls = telescope_sim.get_observation(ptt_actuation, dm_actuate=-Y)
            else:
                # Otherwise, Y is the typical set of PTT actuations fit to the pupil plane
                X, Y, strehls = telescope_sim.get_observation(ptt_actuation - Y)
        
        if render:
            telescope_sim.render(X, strehls)


    print('')
    
    

if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()
    
    # Add all the arguments for Multi-Aperture Telescope 
    parser = add_multi_aperture_telescope_args(parser)
    
    parser.add_argument('--num_steps', type=int,
                        default=20,
                        help='Number of steps to run.')
    
    parser.add_argument('--step_time_granularity', type=float,
                        default=0.01,
                        help='The time granularity of simulation step (seconds)')
    
    parser.add_argument('--add_ptt_perturbations_sigma', type=float,
                        help='Sigma of the normal distribution of errors to add to piston, tip, and tilt each step. (Default: None) (modulated by PTT scales set in config)')
    
    parser.add_argument('--apply_optimal_actuator_corrections', 
                        action='store_true',
                        help='Apply the best fit actuator corrections (PTT or DM)')
    
    parser.add_argument('--no_render', 
                        action='store_true',
                        help='Disable environment render function')
    
    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)
