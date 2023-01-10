"""
Simulate multi-aperture telescope optics with piston, tip, and tilt control using HCIPy

Author: Ian Cunnyngham (Institute for Astronomy, University of Hawai'i) 2019-2023

Features:
 - Sets up multi-aperture pupil plane with tip-tilt piston control
 - Sets up multiple broadband filters consisting of a number of discrete 
   monochromatic simulations around a central wavelength
 - Generates PSFs for all filters/focal-planes for a given piston, tip, tilt, and HCIPy atmosphere
   and optionally...
   - Scale PSFs (by max possible intensity, and/or by exponentiation)
   - Convove an image with PSFs (returned instead of PSFs)
   - Bundle FFTs of PSFs (or convolved images) (real and imaginary) for machine learning
   - Add noise to output samples
 - Measure linear fit of pupil-plane piston, tip, tilt (if atmosphere provided)
 - Return strehl measures
 - Configurable spiders
 
 Updates integrated 2020-11
 - Photon / detector noise via HCIPy's NoisyDetector
 - Deformable Mirror
   - Aproximate PTT via DM for bench demo actuation
   - Propogate DM to PSF to simulate bench demo
   - (TODO) Interface for imposing custom DM actuation and propogating
 - Central obsctruction aperture
 
 2021-03
 - Deformable mirror can now be directly used for actuation instead of PTT actuation, or PTT aproximation via the DM
   (i.e. the DM can be set instead of ptt when generating samples and residual fits of the DM to atmospheres returned)

 2023-01
 - Added per-sample normalization
 
 .sample() returns data structured for ML (more details in function): 
     X: (res,res,samples) tensor of PSFs (or convolved images) + (optionally) FFTs
     Y: (nMir, 3) matrix of piston, tip, tilt (measured if atmosphere passed in)
 
"""

import numpy as np
import hcipy
from scipy.signal import fftconvolve


class MultiAperturePSFSampler:
    """
    Multi-aperture telescope PSF sampler class
    
    Parameters
    - - - - - -
    mirror_config : (dict) mirror configuration details
        {
            'pupil_extent':, ...,  # (float) Spatial extent of pupil plane (meters)
            'pupil_res': ...,      # (int) Resolution of pupil plane
            'positions':           # HCIPy cartesian grid of positions (meters)
            'aperture_config':     # Paramters for each mirror's HCIPy aperture function
                                   # e.g. ['circular', mir_diamater] 
                                   #    -> hcipy.circular_aperture(mir_diamater)  (meters)
            'piston_scale', ...,   # (float) Scale piston actuation (meters)
            'tip_tilt_scale', ...  # (float) Scale of tip-tilt actuation (meters)
                                   #   Note: meausred pupil-plane errors will be 2x larger than segment actuation 
            'dm_config':           # None (no deformable mirror) or list of params passed to DM constructor
            'dm_act_selection'     # (bool array) (Kinda hacky) My DM aperature selection seems to be somewhat 
                                                  non-deterministic, so pass in the DM actuator selection mask
                                                  in order to make sure the same ones are always used. (Optional)
            'aprox_ptt_wih_dm':    # (bool) If True, uses DM to approximate segment piston tip and tilt
            'spider_config':       # None (no spider) or dict of spider wiidth and orientation angle
            {  
                'width':           # (float) width of spider in pupil plane (meters)
                'angle':           # (float) angle of spider orientation (degrees)
                'random_angle':    # (bool) If True, randomize spider angle each initialization
            }
        }
    filter_configs : (list of dicts) specifying filter configuations
        [ {
            'central_lam': ...,    # (float) Central wavelgnth of filter (meters)
            'focal_extent': ...,   # (float) Angular extent of focal plane (arcsec)
            'focal_res': ...,      # (int) resolution of focal-plane PSF generated
            'frac_bandwidth': ..., # (float) fractional bandwidth of filter ( (1 +/- this/2 ) * central_wavelength )
            'num_samples': ...     # (int) Number of monocromatic PSFs across the bandwidth range
            'detector_config':     # (dict) (optional) If set, kwargs passed into NoisyDetector
        }, ...]
    extra_processing: dict of extra steps the sampler can do
       {
           'include_fft': ...,     # (bool) Bundle FFT (real, imag) with PSF samples for machine learning
           'max_inten_norm': ...,  # (bool) Normalize PSFs by max acheivable intensity
           'per_sample_norm': ..., # (bool) Normalize each PSF individually from [0-1]
           'pow_scale': ...,       # (float or False) Scale output by this power 
           'gauss_noise': ...,     # (float or False) sigma of gaussian noise (added after int norm, before power scaling)
       }
    
    """
    def __init__(self, 
                 mirror_config, 
                 filter_configs, 
                 extra_processing=None):
        self.mirror_config = mirror_config
        self.filter_configs = filter_configs
        
        # defaults for extra_processing (could do this with other configs...)
        ep_default = {
           'include_fft': False,
           'max_inten_norm': True,
           'per_sample_norm': False,
           'pow_scale': False,
           'gauss_noise': False   
        }
        if extra_processing is None:
            extra_processing = ep_default
        else:
            # Add in any missing keys
            for key, value in ep_default.items():
                if key not in extra_processing:
                    extra_processing[key] = value
        self.extra_processing = extra_processing
        
        ### Setup HCI
        
        # Create pupil plane
        self.pupil_grid = hcipy.make_pupil_grid(mirror_config['pupil_res'], mirror_config['pupil_extent'])
        
        # Center points for each mirror as hcipy.CartesianGrid
        self.mir_centers = mPos = mirror_config['positions']
        # Count number of mirrors
        self.nMir = mPos.x.shape[0]
        
        # First argument of list selects aperture
        # Further arguments passed directly to aperture constructor
        ## Calculate area for entire aperture to use with photon flux calculation
        ## Note: Not going to go out of my way to account for the spiders
        aper_cfg = mirror_config['aperture_config']
        if aper_cfg[0] == 'circular':
            aper_shape = hcipy.circular_aperture(*aper_cfg[1:])
            D = aper_cfg[1]
            self.aper_area = self.nMir * np.pi*((D/2)**2)
            
        if aper_cfg[0] == 'circular_central_obstruction':
            aper_shape = hcipy.make_obstructed_circular_aperture(*aper_cfg[1:])
            D, cen_obs_ratio = aper_cfg[1:3]
            self.aper_area = self.nMir * np.pi*( ((D/2)**2) - ((D*cen_obs_ratio/2)**2) )
            
        if aper_cfg[0] == 'hexagonal':
            aper_shape = hcipy.hexagonal_aperture(*aper_cfg[1:])
            D = aper_cfg[1]  # Circumdiamater
            self.aper_area = self.nMir * (3*np.sqrt(3)/8) * (D**2)
        
        # Setup pupil-plane aperture and segments
        aper, segments = hcipy.make_segmented_aperture( 
                            aper_shape,  
                            mPos, 
                            return_segments=True)
        self.aper = aper(self.pupil_grid)
        self.segments = hcipy.evaluate_supersampled(segments, self.pupil_grid, 1)
        # Add piston, tip, tilt control to each sub-aperture
        self.sm = hcipy.SegmentedDeformableMirror(self.segments)
        
        ## Setup Deformable Mirror if dm_config is not None
        dm_config = mirror_config.get('dm_config', None)
        # Obviously this will fail if dm_config is not set properly
        self.aprox_ptt_wih_dm = mirror_config.get('aprox_ptt_wih_dm', False)
        if dm_config is not None:
            # Scale input and output DM actuators by this for ML purposes
            # Maybe this should be configurable, but hard coded for now
            self.dm_act_scale = 1e6
            
            # (TODO) Figure out how accurate is model of xinetics influence functions
            self.dm_influence_basis = hcipy.make_xinetics_influence_functions(self.pupil_grid, *dm_config)
            self.dm = hcipy.DeformableMirror(self.dm_influence_basis)
            
            # Create selection mask for pupil-plane pixels that actually contribute
            self.aper_sel = self.aper != 0
            
            # Extract the masked part of each influence funciton
            # Turn into a numpy matrix and transpose (for use in least squares fitting)
            dm_influence_matrix = np.array([basis[self.aper_sel] for basis in self.dm_influence_basis ]).T
            
            # The process of thresholding "active" actuators below is apparently somewhat
            # non-deterministic, to the point I was getting different actuator selections
            # between runs.  So, for now, I am providing the hack of directly saving and loading
            # the actuator selection mask.
            dm_act_selection = mirror_config.get('dm_act_selection', None)
            if dm_act_selection is not None:
                self.act_sel = dm_act_selection
            else:
                # Create a "meta influence function", how much each actuator actually contributes
                # when you mask out the usable aperture
                aper_actuate_infl = dm_influence_matrix.sum(axis=0)

                # Select actuators which contribute at least 2% of the most influential ones 
                # (Experimentally determined number, but the more of the tail end you allow, the more
                #  a least squares fit overfits those actuators at the expense of other ones)
                self.act_sel = aper_actuate_infl > aper_actuate_infl.max()/50
            
            # Apply actuator selection to our influence matrix
            self.dm_influence_matrix = dm_influence_matrix[:, self.act_sel]
        else:
            self.dm = None
        
        ## Generate spiders if not None
        # NOTE: Add Spiders to aperture *AFTER* Deformable mirror, otherwise it influences DM actuator functions     
        # There are HCIPy apertures which implemnt spiders more cleanly, but this is more flexible
        # (I don't believe you can control the spider orientations in the native ones at the moment)
        spider_cfg = mirror_config.get('spider_config', None)
        if spider_cfg is not None:
            s_width = spider_cfg['width']
            s_angle = spider_cfg.get('angle', 0)
            
            # If random_angle set to True, generate random orientation
            if spider_cfg.get('random_angle', False):
                s_angle = np.random.uniform(0, 90)
                
            # Store for retreival 
            self.spider_angle = float(s_angle)
            
            # Convert to radians
            s_angle *= np.pi/180
            
            # Generate spider coordinates for ends
            p_ext = mirror_config['pupil_extent']
            spider1_start = p_ext*np.cos(s_angle), p_ext*np.sin(s_angle)
            spider1_end = p_ext*np.cos(s_angle+np.pi), p_ext*np.sin(s_angle+np.pi)
            spider2_start = p_ext*np.cos(s_angle + np.pi/2), p_ext*np.sin(s_angle + np.pi/2)
            spider2_end = p_ext*np.cos(s_angle+np.pi + np.pi/2), p_ext*np.sin(s_angle+np.pi + np.pi/2)
            
            # Generate HCIPy spiders
            spider1 = hcipy.aperture.generic.make_spider(spider1_start, spider1_end, s_width)
            spider2 = hcipy.aperture.generic.make_spider(spider2_start, spider2_end, s_width)
        if spider_cfg:
            self.aper *= spider1(self.pupil_grid) * spider2(self.pupil_grid)

        # Setup each virtual filter
        # e.g. dichroic splits light into a narrow and wide filter with two detectors
        self.lam_setups = []
        for f_config in filter_configs:
            lam = f_config['central_lam']
            fov = f_config['focal_extent']
            f_res = f_config['focal_res']
            frac_bw = f_config['frac_bandwidth']
            num_samples  = f_config['num_samples']
            detector_cfg = f_config.get('detector_config', None)
            
            # Make focal-plane grid (i.e. detector plane)
            f_grid = hcipy.make_uniform_grid([f_res]*2, fov*np.pi/(180*3600))
            prop = hcipy.FraunhoferPropagator(self.pupil_grid, f_grid)
            
            # If noisy detector is defined for this filter, pass in details as kwargs
            if detector_cfg is not None:
                if 'read_noise' not in detector_cfg:
                    # Error if not set
                    detector_cfg['read_noise'] = 0
                detector = hcipy.optics.NoisyDetector(f_grid, **detector_cfg)
            else:
                detector = None
            
            # Create evenly spaced wavelengths for monochromatic samples
            # across fractional bandwidth of filter
            if num_samples > 1:
                filter_lams = lam * np.linspace(1 - frac_bw / 2., 1 + frac_bw / 2., num_samples)
            else:
                filter_lams = [ lam ]
            
            # For each wavelength make a monochromatic hcipy.Wavefront
            wfs = [ hcipy.Wavefront(self.aper, fil_lam ) for fil_lam in filter_lams ]
            
            lam_setup = {
                'f_grid': f_grid,
                'prop': prop,
                'fil_lams': filter_lams,
                'wfs': wfs,
                'detector': detector,
            }
            
            # Generate PSF with no errors to measure peak brightness for strehl
            ref_psf = self._psf(lam_setup)
            lam_setup['peak_int'] = ref_psf.max()
            lam_setup['peak_ind'] = ref_psf.argmax()
            
            # Store normalization for PSF convolving
            # (The PSF should sum to 1 when convolving)
            lam_setup['ref_psf_sum']  = ref_psf.sum()
            if self.extra_processing['max_inten_norm']:
                lam_setup['ref_psf_sum'] /= lam_setup['peak_int']
            
            self.lam_setups += [ lam_setup ]
        
        # Setup pupil-plane segment coordinates for measuring PTT errors
        self.seg_coords = []
        for s in self.segments:
            inds = s.nonzero()
            xs = self.pupil_grid.x[inds]
            ys = self.pupil_grid.y[inds]
            self.seg_coords += [{
                'inds': inds,
                'xs': xs-xs.mean(),
                'ys': ys-ys.mean(),
                'offset': np.zeros(len(inds[0]))+1
            }]
    
    def _psf(self, lam_setup, atmos=None):
        """
        Generate a single fitler's PSF assuming sub-aperture PTT or DM is already set
        
        Parameters
        - - - - - - 
        lam_setup: (dict) One of the filter setups built in __init__ 
        atmos:       (HCIPy atmosphere) (optional) Applies atmosphere, output PTT measured
               
        Output
        - - - - - - 
        (np.array) (focal_res, focal_res) Point spread function
        
        """
        
        prop = lam_setup['prop']
        wfs = lam_setup['wfs']
        
        # If DM is setup, use DM for actuation
        # Otherwise use sub-aperture PTT control
        # TODO: We will eventually want to simulate both
        if self.dm is not None:
            actuators = self.dm
        else:
            actuators = self.sm
        
        focal_total = 0
        # For each monochromatic Wavefront...
        for wf in wfs:
            if atmos is not None:
                # Apply atmos then actuation
                wf_sm = actuators(atmos(wf))
            else:
                # Apply actuation to pupil plane wf
                wf_sm = actuators(wf)

            # Propagate from SM to image plane
            focal_total += prop(wf_sm).intensity
            
        return focal_total.shaped
        
    def _fft_sample(self, psf):
        """Take PSF's FFT, separate into real and imag components, stack into tensor"""
        psf_fft = np.fft.fft2(psf, norm="ortho")
        return np.stack((psf_fft.real, psf_fft.imag), axis=2)
    
    def _measure_atmos_ptt(self, atmos):
        """Given an HCIPy atmosphere, measure best-fit PTT scaled to mirror config"""
        
        # Sample 1 micron atmosphere
        ref_atmos = atmos.phase_for(1e-6)/(2*np.pi)
        
        fits = np.zeros((self.nMir, 3))
        # Fetch pupil plane indices for each sub-aperture 
        # then perform a least-squares fits for offset as well as x and y slope
        for i_s, sc in enumerate(self.seg_coords):
            inds, off, xs, ys = sc['inds'], sc['offset'], sc['xs'], sc['ys']
            #print(inds[0].shape, off.shape, xs.shape, ys.shape, ref_atmos[inds].shape)
            x, _, _, _ = np.linalg.lstsq(np.vstack([off, xs, ys]).T, ref_atmos[inds], rcond=None)
            fits[i_s] = x
        # Remove the mean piston across all segments (doesn't affect PSF)
        fits[:, 0] -= fits[:, 0].mean()
        # Scale measurements by config scales, then divide by two 
        # because mirror actuation changes the overall path-length by 2X
        fits[:, 0] *= (1e-6)/self.mirror_config['piston_scale']/2
        fits[:, 1:] *= (1e-6)/self.mirror_config['tip_tilt_scale']/2
        return fits
    
    def _aprox_via_dm(self, surface):
        """ 
        Approximate an HCIPy pupil-plane surface (assumed to have the right structure and units)
        - Pass in the segmented mirror (self.sm.surface) and the DM makes an approximation
        - Pass in an atmosphere layer surface and the DM will fit that
        """
        # Cast as numpy array and apply aperture selection mask
        surface = np.array(surface)[self.aper_sel]
        
        # Simply do a least squares of all the influence functions (as masked out in init)
        x, _, _, _ = np.linalg.lstsq(self.dm_influence_matrix, surface, rcond=None)
        
        return x
    
    def sample(self, 
               ptt_actuate=None, 
               dm_actuate=None,
               atmos=None,
               convolve_im=None,
               int_phot_flux=None,
               meas_strehl=False
              ):
        """
        Generate a PSF based on PTT and/or atmosphere, return PSFs or convolved images (X) and best PTT (Y)
        
        Parameters
        - - - - - - 
        ptt_actuate: (ndarray) (optional) (nMir x 3), piston, tip, tilt actuation to impose scaled by 
                     factors set up in mirror_config.  Note: Pupil plane errors are 2x these!
                     If none given, set to zero errors (could be set to differential instead...)
        dm_actuate:  (ndarray) (optional) (entry for each active actuator) dm segment actuation
        atmos:       (HCIPy atmosphere) (optional) Applies atmosphere, output PTT measured
        convolve_im: (ndarray) (optional) the image you want to convolve with the PSF or False return PSF
                     Note: Assumes the pixel scale of the filter PSFs is set for image's angular extent 
                     Convolves for each filter, extra_processing steps will be appleid to conv images (including FFTs)
        int_phot_flux:  (float or array w/ entry for each filter) 
                        Time-integrated photon flux.  photons / m^2
                        Note: detector_config must be setup for each filter
        meas_strehl: (bool) If true, returns the strehls for each feature
                     
        Output
        - - - - - - 
        X: (res,res,samples) tensor of PSFs (or convolved images) + (optionally) FFTs
        Y: (nMir, 3) matrix of piston, tip, tilt (measured if atmosphere passed in)
        
        """
        
        ## If no PTT sent in, assume that means zero actuation, reset SM
        if ptt_actuate is None:
            ptt_actuate = np.zeros((self.nMir, 3))
        if (self.dm is not None) and (dm_actuate is None):
            dm_actuate = np.zeros((self.act_sel.sum(), ))
        
        # Set sub-aperture piston, tip, and tilts
        pist = ptt_actuate[:, 0] * self.mirror_config['piston_scale']
        tts = ptt_actuate[:, 1:] * self.mirror_config['tip_tilt_scale']
        self.sm.set_segment_actuators(np.arange(self.nMir), pist, tts[:, 0], tts[:, 1])
        
        # Aproximate above actuation with DM if set
        if self.aprox_ptt_wih_dm:
            dm_actuate = self.dm_act_scale * self._aprox_via_dm(self.sm.surface)
        
        if self.dm is not None:
            # Apply to relevant actuators in dm
            self.dm.actuators[self.act_sel] = dm_actuate / self.dm_act_scale
            
        if int_phot_flux is not None:
            # Tile to all filters if only one value supplied
            ip_flux = np.array([int_phot_flux]).flatten()
            if ip_flux.shape[0] == 1:
                ip_flux = np.tile(ip_flux, len(self.lam_setups))

        # Build up samples (if multiple filters) to stack into tensor for prediction
        Xs = []
        
        # And measured strehls if requested
        strehls = []
        
        # For each filter...
        for i_filter, lam_setup in enumerate(self.lam_setups):
            
            # Simulate filter's PSF
            psf = self._psf(lam_setup, atmos=atmos)
            
            # Measure strehl for fitler if requested
            if meas_strehl:
                strehls += [ psf.flat[ lam_setup['peak_ind'] ] / lam_setup['peak_int'] ]
            
            # Normalize PSF is set
            if self.extra_processing['max_inten_norm']:
                psf /= lam_setup['peak_int']
            
            if convolve_im is not None:
                # If image passed in, convolve it with normalized PSF as output
                out_samp = fftconvolve(convolve_im, psf/lam_setup['ref_psf_sum'], mode='same')
            else:
                # Otherwise output is the PSF itself
                out_samp = psf
            
            # Probably remove this for better detector noise
            if self.extra_processing['gauss_noise'] is not False:
                out_samp += np.abs(np.random.normal(0, self.extra_processing['gauss_noise'], psf.shape))
            
            ### Create detector noise
            # If integrated photon flux isn't provided, not sure how much sense
            # read and other noise sources make, so only apply detector when it is provided
            if (lam_setup['detector'] is not None) and (int_phot_flux is not None):
                cur_ipf = ip_flux[i_filter]
                out_samp = self._addNoiseToObservation(
                    observation=out_samp, 
                    int_phot_flux=cur_ipf, 
                    i_filter=i_filter
                )
            
            # Scale output for DNN consumption via power law
            # (Small fractional powers compress range and gets ouput closer to standarized)
            if self.extra_processing['pow_scale'] is not False:
                out_samp = np.power(out_samp, self.extra_processing['pow_scale']) 
    
            if self.extra_processing['per_sample_norm'] is not False:
                samp_min, samp_max = out_samp.min(), out_samp.max()
                out_samp = (out_samp - samp_min)/(samp_max - samp_min)
    
            Xs += [ out_samp[..., None] ]
            if self.extra_processing['include_fft']:
                # If set, add FFT of this filter's output as extra channels
                Xs += [ self._fft_sample(out_samp) ]
        
        # Return PTT that best phases sub-apertures
        if (self.dm is not None) and (not self.aprox_ptt_wih_dm):
            out_actuate = np.copy(dm_actuate)
            if atmos is not None:
                atmos_surface = atmos.phase_for(1)/(4*np.pi)
                atmos_surface -= atmos_surface.mean()
                out_actuate += self.dm_act_scale * self._aprox_via_dm(atmos_surface)
        else:
            out_actuate = np.copy(ptt_actuate)
            if atmos is not None:
                # If atmosphere is provided, compute the best fit piston, tip, and tilts in addition
                out_actuate += self._measure_atmos_ptt(atmos)
        
        # Combine list of X samples into tensor
        Xs = np.concatenate(Xs, axis=2)
        if meas_strehl:
            return Xs, out_actuate, strehls
        else:
            return Xs, out_actuate
    
    def _addNoiseToObservation(self,
        observation,    # Assumed to be PSF or extended image of the right shape for detector
        int_phot_flux,  # Time-integrated photon flux for a single filter.  photons / m^2 
        i_filter=0,     # When sampler is setup to return multiple filters, need to specify which filter index
    ):
        """Simulate noise for a given observation.  Note detector must be setup.  Broken out to allow evolving atmos"""
        
        # retrieve current filter setup
        lam_setup = self.lam_setups[ i_filter ]
        wf = lam_setup['wfs'][0]

        wf = hcipy.Wavefront(
            hcipy.Field(
                np.sqrt(observation.flatten()).astype("complex128"), 
                wf.grid
            ), 
            wf.wavelength
        )
        
        ## Note: It appears when this is passed in, the scaling of
        ##       the PSF is irrelevant.
        # num_photons/m^2 -> num_photons
        wf.total_power =  int_phot_flux * self.aper_area
        
        detector = lam_setup['detector']
        
        # Because we're dealing with static atmospheres, integrations above a certain amount
        # of time don't make any sense, so instead of passing additional "integration time" param into this 
        # function, photon flux is assumed already integrated, and we passholder "integrate" here for "1 second"
        # This will be different if dark current is taken into account
        detector.integrate(wf, 1)
        
        # "Read out" detector, convert hcipy Field back into numpy array
        read_out = detector.read_out()
        
        # Return observation in 2d rather than flattned shape.  
        # Shouldn't need to do abs, but poisson read noise seems to allow negative for some reason
        return np.array(np.abs(read_out.shaped))

    
    def getPhaseScreen(self, 
                       atmos=None, 
                       filter_id=0,
                       np_array=False
                      ):
        """
        Return a monochromatic phase screen for a single filter
        
        In most cases, we'll probably be simulating poly-chromatic filters by combining at least
        three monochromatic pupil-planes for each, but for visualization purposes we are really only 
        interested in the phase screen of one of these pupil-planes.  Right now it picks the
        left-most monochromatic phase screen (because it would take slightly more effort to return
        the midddle), but this is probably good enough for visualization purposes.
        
        Parameters
        - - - - - - 
        atmos:       (HCIPy atmosphere) (optional) Applies atmosphere to pupil-plane
                     (Note: atmosphere not currently stored in sampler, but previous PTT state is)
        filter_id:   (integer) Which filter (corresponding to filter_setups index) to return phase screen
        np_array:    (bool) By default, return a HCIPy Field, which stores a bunch of 
                            information besides the raw values (like pupil-plane coordinates)
                            and has a funky structure, if we just want the raw phase values
                            in a numpy array, set this to True
                     
        Output
        - - - - - - 
        phase_screen: (res,res) monochromatic phase-screen with domain (-pi, pi)
        
        """
        
        # Grab one monochromatic "wavefront" from selected filter
        wf = self.lam_setups[filter_id]['wfs'][0]
        
        # Apply current actuation to wavefront
        if self.dm is not None:
            actuators = self.dm
        else:
            actuators = self.sm
        pp = actuators(wf)
        
        # If atmosphere passed in, apply to current pupil-plane
        if atmos is not None:
            pp = atmos(pp)
        
        # Grab the phase of the wavefront
        phase = pp.phase
        
        if np_array:    # If we want a raw numpy array
            # Zero out all values outside the aperture
            phase = phase*self.aper
            
            # Reshape to the same shape as the pupil plane and return
            return phase.reshape(self.pupil_grid.shape)
        else:
            # Otherwise, return the HCIPy Field, which is a little nicer to plot
            return phase
        
    def _from_mag_gen_rand_vec(self, mag):
        """Generate randomly oriented vector. Given a scaler (float) generate a random direction"""
        th = np.random.uniform(0, 2*np.pi)
        return mag*np.cos(th), mag*np.sin(th)
    
