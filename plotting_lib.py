# Import necessary libraries
import datetime
import sys
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import pyfesom2 as pf
from matplotlib.colors import TwoSlopeNorm

# Recent changes:
# Adjust variable names in fesom.mesh.diag.nc for compatibility with most recent FESOM version
# Change plotting from a file in resultpath to plotting directly from a dobj_data

def plot(
	dobj_data,
	meshdiag: xr.Dataset,
	meshpath: str,
	str_id: str,
	time: int, 
	level: int,
	filter_lat: bool = False, lat_north: float = 12.7525, lat_south: float = 5.0071, 
	unglue=True, cyclic_length=4.5,
):

	"""
	Plot data of the FESOM model output for the Soufflet channel 
	for a given time and vertical level.

	Parameters:
	- dobj_data (xr.DataArray or xr.Dataset): data file which should be plotted
	- meshdiag (xr.Dataset): xr.open_dataset(fesom.mesh.diag.nc)
	- meshpath (str): Path to the mesh files (nod2.out, elem2d.out etc.).
	- str_id (str): Variable name to plot.
	- time (int): Time index to select (-1 for last).
	- level (int): Vertical level index to select (if data has a vertical dimension).
	- filter_lat (bool, default: False): Whether to filter data based on latitude (most eddy activity seems to happen between lat_north and lat_south)
	- lat_north (float, optional): Latitude to cut off data to the north.
	- lat_south (float, optional): Latitude to cut off data to the south.
	- unglue (bool): Whether to unglue the data at the periodic boundaries. Without ungluing,
	there can be some spurious meridional patterns.
	- cyclic_length (float): Width of the channel in degrees for ungluing.
	"""

	# Set rotation angles
	alpha, beta, gamma = 0, 0, 0

	# Infer meshpath if not provided
	if not meshpath:
		inferred_meshpath = os.path.join(resultpath, '../')  # Default inference logic (one level up)
		meshpath = inferred_meshpath

	# Load mesh
	print("Loading mesh...")
	mesh = pf.load_mesh(meshpath, abg=[alpha, beta, gamma], usepickle=False)

	# Calculate mean X and Y coordinates for each element
	elem_n = meshdiag.elem.shape[0]
	xx2 = mesh.x2[mesh.elem[:, :elem_n]].mean(axis=1)
	yy2 = mesh.y2[mesh.elem[:, :elem_n]].mean(axis=1)

	# Load dataset and select time
	if isinstance(dobj_data, xr.Dataset):
		dat = dobj_data[str_id]
	else:
		dat = dobj_data
	
	dat = dat.isel(time=time)

	# Check if data has a vertical dimension and select level if present
	if 'nz1' in dat.dims or 'nz' in dat.dims:
		dat = dat.isel(nz1=level) if 'nz1' in dat.dims else dat.isel(nz=level, missing_dims="ignore")
	dat = dat.squeeze()  # Remove extra dimensions

	# Determine plot coordinates
	if "nod2" in dat.dims:
		print("nod2 found in dat.dims, setting X, Y = (meshdiag.lon, meshdiag.lat)")
		X, Y = (meshdiag.lon, meshdiag.lat)
	else:
		print("nod2 not found in dat.dims, setting X, Y = (xx2, yy2) (averaged meshdiag.lon and .lat to triangle centers")
		X, Y = (xx2, yy2)
	print("Shape of X, Y, dat:", X.shape, Y.shape, dat.shape)

	# Filter data based on latitude limits, if specified
	if filter_lat:
		print("Filtering latitudes...")
		# Get the latitude array from the mesh diagnostics or data
		lat_data = meshdiag.lat if 'nod2' in dat.dims else yy2

		# Create a mask for filtering based on latitude
		lat_mask = (lat_data >= lat_south) & (lat_data <= lat_north)

		# Convert lat_mask to an xarray DataArray if using 'nod2' coordinates
		if 'nod2' in dat.dims:
			lat_mask = xr.DataArray(lat_mask, dims=["nod2"], coords={"nod2": dat["nod2"]})
		else:
			lat_mask = xr.DataArray(lat_mask, dims=["elem"], coords={"elem": dat["elem"]})

		# Apply mask to dat, and filter X and Y using a plain NumPy mask
		dat = dat.where(lat_mask, drop=True)
		X = X[lat_mask.values]
		Y = Y[lat_mask.values]

	 # Unglue mesh if required
	if unglue and str_id in ["u", "v"]: # No need for ungluing for unod, vnod, curl_u, w
		try:
			# Load triangulation data and node coordinates
			tri = np.loadtxt(f'{meshpath}elem2d.out', skiprows=1, dtype=int)
			nodes = np.loadtxt(f'{meshpath}nod2d.out', skiprows=1)
			xcoord, ycoord = nodes[:, 1], nodes[:, 2]
		
			# Map x and y coordinates to triangles
			xc, yc = xcoord[tri - 1], ycoord[tri - 1]
		
			# Adjust cyclic coordinates
			xmin = xc.min(axis=1)
			for i in range(3):
				ai = np.where(xc[:, i] - xmin > cyclic_length / 2)
				xc[ai, i] -= cyclic_length
		
			# Set X and Y as the mean along the second axis
			X = xc.mean(axis=1)
			Y = yc.mean(axis=1)
		except FileNotFoundError:
			print("Required files for ungluing not found; skipping unglue step.")
			
	# Set units and colormap based on units
	units = getattr(dat, 'units', 'none')
	colormap_info = {
		'm/s': (cmocean.cm.balance, 'balance'),
		'1/s': (cmocean.cm.balance, 'balance'),
		'C': (cmocean.cm.thermal, 'thermal'),
		'm': ('Greys', 'greyscale')
	}
	cmap, cmap_type = colormap_info.get(units, (cmocean.cm.balance, 'balance'))

	# Determine color limits with custom rounding using np.round
	dat_min, dat_max = dat.min().values, dat.max().values
	if units == 'm':
		vmin, vmax = int(np.round(dat_min)), int(np.round(dat_max))
	elif units in {'m/s', '1/s'}:
		max_val = max(abs(dat_min), abs(dat_max))
		max_val = np.round(max_val, 7) if max_val < 1e-5 else np.round(max_val, 5)
		vmin, vmax = -max_val, max_val
	elif units == 'C':
		vmin, vmax = np.round(dat_min, 1), np.round(dat_max, 1)
	else:
		vmin, vmax = dat_min, dat_max  # default behavior if units are unknown

	# Set up figure
	fig, ax = plt.subplots(figsize=(5, 20))

	# Initialize norm for TwoSlopeNorm only when required
	norm = None
	if units in {'m/s', '1/s', 'none'}:
		norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

	print(X.shape, Y.shape, dat.shape)
	
	# Apply tripcolor without vmin/vmax if norm is defined
	im = ax.tripcolor(X, Y, dat, shading='flat', cmap=cmap, norm=norm, 
					  **({'vmin': vmin, 'vmax': vmax} if norm is None else {}))

	# Configure plot aesthetics
	ax.tick_params(axis='both', labelsize=10)  # Set tick label size
	ax.set_xlabel('lon / deg', size=10)  # Set label size
	ax.set_ylabel('lat / deg', size=10)  # Set label size

	# Add colorbar with reduced padding
	cbartext = f'{str_id} / {units}'
	cbar = fig.colorbar(im, orientation='horizontal', pad=0.03, extend='both')  # Adjusted pad
	cbar.set_label(cbartext, size=10)
	cbar.ax.tick_params(labelsize=10)

	# Set title, handling missing attributes gracefully, positioned inside the plot

	def remove_time(datetime=None) -> np.datetime64:
		"""
		Removes the time component from a datetime object, returning only the date.

		Parameters:
			datetime (np.datetime64): A datetime object with both date and time information.

		Returns:
			np.datetime64: The date portion of the input datetime, with the time set to midnight (00:00).

		Example:
			>>> remove_time(np.datetime64('2024-11-05T14:23'))
			np.datetime64('2024-11-05')
		"""
		return datetime.astype('datetime64[D]')
	
	if "nz1" in dobj_data.dims:
		title_text = f'{remove_time(dat.time.values)}, (level,nz1)=({level},{np.round(dat.nz1.values,1)}m)'
	elif "nz" in dobj_data.dims:
		title_text = f'{remove_time(dat.time.values)}, (level,nz1)=({level},{np.round(dat.nz.values,1)}m)'
	plt.title(title_text, loc='center', pad=20, fontsize=14, color='black', y=0.95)

	plt.show(block=False)
