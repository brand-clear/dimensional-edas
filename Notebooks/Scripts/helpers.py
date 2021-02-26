# -*- coding: utf-8 -*-
"""
Helper functions for the statistical analysis of inspection data.

"""


import os
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_ROOT = 'Q:\\Quality Control\\quality_controller\\data'
INSPECTION_PATH = os.path.join(DATA_ROOT, 'inspection_data')  # Inspection data
REJECTION_PATH = os.path.join(DATA_ROOT, 'rejected.csv')      # Rejection data


class Params():
	"""Parameters for plots that do not comform to rcParams."""
	FIGURE_WIDTH = 10
	FIGURE_HEIGHT = 10


# Set global pandas/matplotlib parameters
pd.set_option('mode.chained_assignment', None)
plt.rcParams.update({
	'axes.titlepad': 20,
	'axes.grid': True,
	'axes.labelpad': 20,
	'font.size': 16,
	'figure.figsize': (Params.FIGURE_WIDTH, Params.FIGURE_HEIGHT)
})


def set_figure_font_size(size):
	plt.rcParams.update({'font.size': size})


def set_figure_size(width, height):
	plt.rcParams['figure.figsize'] = (width, height)
	Params.FIGURE_WIDTH = width
	Params.FIGURE_HEIGHT = height




class FeatureDataFrame(ABC):
	"""Base class for DataFrames of a specific feature type.


	Parameters
	----------
	init : bool
		Specifies if the FeatureDataFrame is to be initialized (created from 
		scratch), optional. The default is True.

	Attributes
	----------
	CONTROLS : dict
		Defines the table structure of the DataFrame. To be implemented by subclass.
	PICKLE_FILE : str
		Defines the name of the pickle file that contains the DataFrame. To be 
		implemented by subclass.
	extract_from_dataframe : callable
		Defines how the FeatureDataFrame extracts new features from a 
		DataFrame. To be implemented by subclass.
	df : DataFrame
		The underlying data source.
	existing_ids : list

	"""
	
	@property
	@abstractmethod
	def CONTROLS(self):
		pass
	

	@property
	@abstractmethod
	def PICKLE_FILE(self):
		pass


	_PICKLE_DIR = 'C:\\Users\\mcclbra\\Desktop\\Dimensional-Analyses\\Models'


	def __init__(self, init=True):
		self._PICKLE_PATH = os.path.join(self._PICKLE_DIR, self.PICKLE_FILE)
		if init:
			# Create from scratch
			self.df = pd.DataFrame(columns=self.CONTROLS.keys())
		else:
			# Get from file
			self.df = self.deserialize()

				
	@abstractmethod
	def extract_from_dataframe(self, filename, df):
		pass
	

	@property
	def existing_ids(self):
		"""list: Inspection IDs in the DataFrame."""
		return self.df.inspection_id.unique().tolist()
	
	
	def serialize(self):
		"""Save the DataFrame to file."""
		self.df.to_pickle(self._PICKLE_PATH)


	def deserialize(self):
		"""Returns the existing FeatureDataFrame from file."""
		return pd.read_pickle(self._PICKLE_PATH)
 

	def get_part_id(self, inspection_id):
		"""The part id is contained within the inspection ID by design.
		
	
		Parameters
		----------
		inspection_id : str
			An inspection ID.
	
		Returns
		-------
		str or np.nan
	
		"""
		
		try:
			# Option 1: ID with '-' delimiter
			id_split = inspection_id.split('-')
			if id_split[0].isdigit():
				return id_split[1]
			
			# Option 1: ID with '_' delimiter
			id_split = inspection_id.split('_')
			if id_split[0].isdigit():
				return id_split[1]
			
		except IndexError:
			return np.nan
		
		return np.nan


	def get_check_count(self, inspection_id):
		"""The check count represents the number of times a part was inspected 
		by QC.
		
		
		Parameters
		----------
		inspection_id : str
			An inspection id. The digits that follow '_QC' in the inspection id 
			represent the check count.
	
		Returns
		-------
		int
			
		Notes
		-----
		If the check count cannot be determined, it is assumed that this is the
		first inspection.
	
		"""
		
		id_sans_ext = os.path.splitext(inspection_id)[0]
		index = id_sans_ext.find('_QC')
		if index != -1:
			return int(id_sans_ext[index + 3:])

		return 1




class PlaneDataFrame(FeatureDataFrame):
	"""A DataFrame that contains plane features only."""
	

	CONTROLS = {'inspection_id': np.nan, 'feature': np.nan, 
				'x_nom': np.nan, 'x_meas': np.nan, 'x_dev': np.nan,
				'y_nom': np.nan, 'y_meas': np.nan, 'y_dev': np.nan, 
				'z_nom': np.nan, 'z_meas': np.nan, 'z_dev': np.nan, 
				'total_runout_ab': np.nan, 'flatness': np.nan, 
				'profile': np.nan, 
				'part_id': np.nan, 'check_count': np.nan}


	PICKLE_FILE = 'planes.pkl'


	def __init__(self, init=True):
		super().__init__(init)


	def extract_from_dataframe(self, filename, df):
		"""      
		Parameters
		----------
		filename : str
			The name of the CSV file containing the DataFrame.
		df : DataFrame
			A DataFrame containing inspection data.
	
		Returns
		-------
		records : list
			A collection of dictionaries that represent plane features found in 
			the DataFrame.
	
		"""
		
		records = []
		planes = [i for i in df.Name.unique() if 'plane' in i]
		part_id = self.get_part_id(filename)
		check_count = self.get_check_count(filename)
	
		for p in planes:
			
			# Isolate feature in DataFrame
			p_df = df[df.Name == p]
			
			controls = self.CONTROLS.copy()
			controls['inspection_id'] = filename
			controls['part_id'] = part_id
			controls['check_count'] = check_count
			controls['feature'] = p
			
			# Get x, y, z values
			try:
				controls['x_nom'] = p_df[p_df.Control == 'Centroid X'].Nom.values[0]
				controls['x_meas'] = p_df[p_df.Control == 'Centroid X'].Meas.values[0]
				controls['x_dev'] = controls['x_meas'] - controls['x_nom']
			except IndexError:
				pass
				
			try:
				controls['y_nom'] = p_df[p_df.Control == 'Centroid Y'].Nom.values[0]
				controls['y_meas'] = p_df[p_df.Control == 'Centroid Y'].Meas.values[0]
				controls['y_dev'] = controls['y_meas'] - controls['y_nom']
			except IndexError:
				pass
					
			try:
				controls['z_nom'] = p_df[p_df.Control == 'Centroid Z'].Nom.values[0]
				controls['z_meas'] = p_df[p_df.Control == 'Centroid Z'].Meas.values[0]   
				controls['z_dev'] = controls['z_meas'] - controls['z_nom']
			except IndexError:
				pass
			
			# Get total runout value
			try:
				controls['total_runout_ab'] = p_df[p_df.Control == 'Total Runout A B'].Meas.values[0]
			except IndexError:
				pass
				
			# Get flatness value
			try:
				controls['flatness'] = p_df[p_df.Control == 'Flatness'].Meas.values[0]
			except IndexError:
				pass         
	
			# Get profile value
			try:
				controls['profile'] = p_df[p_df.Control == 'Surface Profile'].Meas.values[0]
			except IndexError:
				pass  
				
			records.append(controls)
			
		return records




class CylinderDataFrame(FeatureDataFrame):
	"""A DataFrame that contains cylinder features only."""


	CONTROLS = {'inspection_id': np.nan, 'feature': np.nan, 
				'perpendicular_a': np.nan, 'total_runout_ab': np.nan, 
				'position': np.nan, 'cylindricity': np.nan, 
				'diameter_nom': np.nan, 'diameter_meas': np.nan, 
				'diameter_dev': np.nan, 'egg_rate': np.nan, 
				'length_nom': np.nan, 'length_meas': np.nan, 
				'length_dev': np.nan, 'egg_rate': np.nan,
				'part_id': np.nan, 'check_count':  np.nan}
	

	PICKLE_FILE = 'cylinders.pkl'


	def __init__(self, init=True):
		super().__init__(init)


	def extract_from_dataframe(self, filename, df):
		"""
		Parameters
		----------
		filename : str
			The name of the CSV file containing the DataFrame.
		df : DataFrame
			A DataFrame containing inspection data.
	
		Returns
		-------
		records : list
			A collection of dictionaries that represent cylinder features 
			found in the DataFrame.
	
		"""
		
		records = []
		cylinders = [i for i in df.Name.unique() if 'cylinder' in i]
		part_id = self.get_part_id(filename)
		check_count = self.get_check_count(filename)

		for c in cylinders:
			
			# Isolate feature in DataFrame
			c_df = df[df.Name == c]
			
			controls = self.CONTROLS.copy()
			controls['inspection_id'] = filename
			controls['part_id'] = part_id
			controls['check_count'] = check_count
			controls['feature'] = c
			
			# Get perpendicularity value
			try:
				controls['perpendicular_a'] = c_df[c_df.Control == 'Perpendicularity A'].Meas.values[0]
			except IndexError:
				pass
				
			# Get total runout value
			try:
				controls['total_runout_ab'] = c_df[c_df.Control == 'Total Runout A B'].Meas.values[0]
			except IndexError:
				pass
			
			# Get position value
			try:
				controls['position'] = c_df[c_df.Control == 'Position A B'].Meas.values[0]
				
			except IndexError:
				
				try:
					controls['position'] = c_df[c_df.Control == 'Position'].Meas.values[0]
				except IndexError:
					pass
				
			# Get cylindricity value
			try:
				controls['cylindricity'] = c_df[c_df.Control == 'Cylindricity'].Meas.values[0]
			except IndexError:
				pass
	
			# Get diameter value
			try:
				controls['diameter_nom'] = c_df[c_df.Control == 'Diameter'].Nom.values[0]
				controls['diameter_meas'] = c_df[c_df.Control == 'Diameter'].Meas.values[0]
				controls['diameter_dev'] = controls['diameter_meas'] - controls['diameter_nom']
			except IndexError:
				pass
	
			# Get length value
			try:
				controls['length_nom'] = c_df[c_df.Control == 'Length'].Nom.values[0]
				controls['length_meas'] = c_df[c_df.Control == 'Length'].Meas.values[0]
				controls['length_dev'] = controls['length_meas'] - controls['length_nom']
	
			except IndexError:
				pass

			# Get egg rate value
			try:
				controls['egg_rate'] = controls['cylindricity'] * 2 / controls['diameter_meas']
			except TypeError:
				pass

			records.append(controls)
			
		return records




class PointDataFrame(FeatureDataFrame):
	"""A DataFrame that contains point features only."""

	
	CONTROLS = {'inspection_id': np.nan, 'feature': np.nan, 
				'x_nom': np.nan, 'x_meas': np.nan, 'x_dev': np.nan,
				'y_nom': np.nan, 'y_meas': np.nan, 'y_dev': np.nan, 
				'z_nom': np.nan, 'z_meas': np.nan, 'z_dev': np.nan,
				'part_id': np.nan, 'check_count': np.nan}


	PICKLE_FILE = 'points.pkl'


	def __init__(self, init=True):
		super().__init__(init)


	def extract_from_dataframe(self, filename, df):
		"""
		Parameters
		----------
		filename : str
			The name of the CSV file containing the DataFrame.
		df : DataFrame
			A DataFrame containing inspection data.
	
		Returns
		-------
		records : list
			A collection of dictionaries that represent point features found in 
			the DataFrame.
	
		"""
		
		records = []
		points = [i for i in df.Name.unique() if 'point' in i]
		part_id = self.get_part_id(filename)
		check_count = self.get_check_count(filename)
		
		for p in points:
			
			# Isolate feature in DataFrame
			p_df = df[df.Name == p]
			
			controls = self.CONTROLS.copy()
			controls['inspection_id'] = filename
			controls['part_id'] = part_id
			controls['check_count'] = check_count
			controls['feature'] = p
			
			# Get x, y, z values
			try:
				controls['x_nom'] = p_df[p_df.Control == 'X'].Nom.values[0]
				controls['x_meas'] = p_df[p_df.Control == 'X'].Meas.values[0]
				controls['x_dev'] = controls['x_meas'] - controls['x_nom']
			except IndexError:
				pass
	
			try:
				controls['y_nom'] = p_df[p_df.Control == 'Y'].Nom.values[0]
				controls['y_meas'] = p_df[p_df.Control == 'Y'].Meas.values[0]
				controls['y_dev'] = controls['y_meas'] - controls['y_nom']
			except IndexError:
				pass
	
			try:
				controls['z_nom'] = p_df[p_df.Control == 'Z'].Nom.values[0]
				controls['z_meas'] = p_df[p_df.Control == 'Z'].Meas.values[0]
				controls['z_dev'] = controls['z_meas'] - controls['z_nom']
			except IndexError:
				pass
	
			records.append(controls)

		return records




class InspectionDatabase():
	"""Represents a database of new part inspection measurements.

	Within the database there are (3) FeatureDataFrames that each hold a 
	distinct feature type. The inspection_id column serves as the primary key 
	that links these (3) FeatureDataFrames together.

	
	Parameters
	----------
	init : bool
		Specifies if the FeatureDataFrames are to be initialized (created from 
		scratch), optional. The default is False.
	update : bool
		Specifies if the FeatureDataFrames are to be updated after 
		deserialization, optional. The default is True.

	"""

	_DATUM_MAP = {'A': 'datum plane A', 'B': 'datum cylinder B', 
				  'C': 'datum cylinder C', 'midplane': 'alignment midplane', 
				  'point': 'alignment point'}
	

	def __init__(self, init=False, update=True):
		if init:
			# Create from scratch
			self.init()
		else:
			# Get from file
			self.retrieve(update)


	@property
	def rejected_ids(self):
		"""list: A collection of rejected inspection IDs."""
		rej_df = pd.read_csv(REJECTION_PATH)
		ids = rej_df.Drawing.tolist()
		return [str(i).replace('.pdf', '.csv') for i in ids]


	def get_inspection_df(self, filepath):
		"""Read and format a DataFrame containing inspection data.
		

		Parameters
		----------
		filepath : str
			Absolute path to CSV file containing inspection data.

		Returns
		-------
		df : DataFrame
			An inspection DataFrame in standard form.
			
		Raises
		------
		pd.errors.EmptyDataError
			CSV file is empty.

		"""
		
		# Load DataFrame and set column names
		df = pd.read_csv(filepath, encoding="ISO-8859-1")
		df.columns = [i for i in df.iloc[4].values]
		df = df.iloc[5:]
		
		# Delete unnecessary columns
		df.drop(columns=['Test', 'Tol', 'Dev', 'Out Tol'], inplace=True)
		
		# Set column datatypes
		df.Nom = pd.to_numeric(df.Nom)
		df.Meas = pd.to_numeric(df.Meas)
			
		return df

		
	def get_new_features(self, existing_ids=[]):
		"""Add missing features to the FeatureDataFrames.
	
	
		Parameters
		----------
		existing_ids : list, optional
			Inspection IDs that already exist in the FeatureDataFrames. 
			The default is [].

		"""
		
		planes = []
		cylinders = []
		points = []
		new_ids = [f for f in os.listdir(INSPECTION_PATH) if f not in 			existing_ids]
		
		# Extract new features
		for f in new_ids:
			try:
				df = self.get_inspection_df(os.path.join(INSPECTION_PATH, f))
			except pd.errors.EmptyDataError:
				continue
			planes.extend(self.planes.extract_from_dataframe(f, df))
			cylinders.extend(self.cylinders.extract_from_dataframe(f, df))
			points.extend(self.points.extract_from_dataframe(f, df))

		# Update FeatureDataFrames
		self.planes.df = self.planes.df.append(planes)
		self.cylinders.df = self.cylinders.df.append(cylinders)
		self.points.df = self.points.df.append(points)
			

	def init(self):
		"""Initialize and serialize FeatureDataFrames."""
	
		# Initialize
		self.planes = PlaneDataFrame()
		self.cylinders = CylinderDataFrame()
		self.points = PointDataFrame()

		# Add feature records
		self.get_new_features()

		# Save to file
		self.planes.serialize()
		self.cylinders.serialize()
		self.points.serialize()


	def retrieve(self, update=True):
		"""Instantiate the serialized FeatureDataFrames.


		Parameters
		----------
		update : bool
			Specifies if the FeatureDataFrames will be updated after retrieval, 
			optional. The default is True.
		
		"""

		self.planes = PlaneDataFrame(False)
		self.cylinders = CylinderDataFrame(False)
		self.points = PointDataFrame(False)

		if update:
			self.get_new_features()


	def part_ids_only(self, ids, df):
		"""Get a DataFrame subset that contains specified part IDs only.
		
	
		Parameters
		----------
		ids : list
			A collection of part IDs.
		df : DataFrame
			The data source.
	
		Returns
		-------
		DataFrame
	
		"""
		
		return df[df.part_id.isin(ids)]


	def parts_only_set(self, part_ids):
		"""Get a subset of each FeatureDataFrame that contains the specified 
		part ids only.
		

		Parameters
		----------
		part_ids : list
			A collection of part ids.

		Returns
		-------
		Tuple

		"""

		return (self.part_ids_only(part_ids, self.planes.df),
				self.part_ids_only(part_ids, self.cylinders.df),
				self.part_ids_only(part_ids, self.points.df))


	def save_all(self):
		"""Serialize all FeatureDataFrame sources."""
		self.planes.serialize()
		self.cylinders.serialize()
		self.points.serialize()


	def drop_rejections(self, df):
		"""Remove all rejected parts from a DataFrame.
		
	
		Parameters
		----------
		df : DataFrame
			The data source.
	
		Returns
		-------
		DataFrame
			A DataFrame subset with approved parts only.
	
		"""
		
		return df[~df.inspection_id.isin(self.rejected_ids)]


	def rejections_only(self, df):
		"""Get a DataFrame subset that contains only rejected part features.
		
		
		Parameters
		----------
		df : DataFrame
			The data source.
	
		Returns
		-------
		DataFrame
	
		"""
		
		return df[df.inspection_id.isin(self.rejected_ids)]


	def datum_only(self, id, df):
		"""Get a DataFrame subset that contains only the datum features.
		
	
		Parameters
		----------
		id : str
			A _DATUM_MAP key.
		df : None or DataFrame
			The data source.
	
		Returns
		-------
		DataFrame
	
		"""
		
		return df[df.feature == self._DATUM_MAP[id]]


	def drop_datums(self, df):
		"""Remove all datum features from a DataFrame.
		
	
		Parameters
		----------
		df : DataFrame
			The data source.
	
		Returns
		-------
		DataFrame
			A DataFrame subset without datum features.
	
		"""
		
		return df[~df.feature.isin(list(self._DATUM_MAP.values()))]


	def print_metadata(self, df, appr_subset=None):
		"""Print high-level stats to the console.

		Parameters
		----------
		df : DataFrame
			The data source.
		appr_subset : DataFrame or None
			A subset of the data source that was approved by QC, optional. The 
			default is None.

		"""

		total_count = len(df.inspection_id.unique())
		print('Number of parts:', total_count)

		if appr_subset is not None:
			appr_count = len(appr_subset.inspection_id.unique())
			percentage = round(appr_count / total_count * 100, 3)
			print('Acceptance rate:', percentage, '%')


	def print_test_case(self, df, feature, upper_bound, lower_bound):
		"""Print a simple test case to the console.


		Parameters
		----------
		df : DataFrame
			The data source.
		feature : str
			A DataFrame column name.
		upper_bound : float
			The max tolerance for the test case.
		lower_bound : float
			The min tolerance for the test case.

		"""
		if upper_bound == .0000 and lower_bound == .0000:
			return

		df = df[~df[feature].isna()]

		passed = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
		percentage = round(len(passed.index) / len(df.index) * 100, 3)
		
		# Set signs
		upper_sign = '+' if upper_bound >= .0000 else ''
		if lower_bound < .0000:
			lower_sign = ''
		elif lower_bound == .0000:
			lower_sign = '-'
		else:
			lower_sign = '+'

		print('Tolerance:', 
			'%s%s / %s%s' % (upper_sign, upper_bound, lower_sign, lower_bound))
		print('Conformity:', percentage, '%')


	def hist(self, df, feature, title='Title', xlabel=None, 
		include_decision=True, kde=True, kde_appr=True, kde_rej=False, binwidth=.0001, upper_bound=.0000, lower_bound=.0000, 
		show_bounds=False):
		"""A histogram that allows for test cases.


		Parameters
		----------
		df : DataFrame
			The data source.
		feature : str
			A DataFrame column name.
		title : str
			The histogram title, optional. The default is 'Title'.
		xlabel : str or None
			The X axis label, optional. The default is None.
		include_decision : bool
			Specifies if the DataFrame is to be split by QC decision, optional. 
			The default is True.
		kde : bool
			Specifies if KDE is applied to the histogram when decision is not 
			included, optional. The default is True.
		kde_appr : bool
			Specifies if KDE is applied to the approved DataFrame subset, 
			optional. The default is True
		kde_rej : bool
			Specifies if KDE is applied to the rejected DataFrame subset, 
			optional. The default is False.
		binwidth : float
			The bin width, optional. The default is .0001.
		upper_bound : float
			The max tolerance for the test case, optional. The default is .0000.
		lower_bound : float
			The min tolerance for the test case, optional. The default is .0000.
		show_bounds : bool
			Specifies if vertical lines are used to represent boundaries.

		"""
		df = df[~df[feature].isna()]

		if include_decision:
			# Split data by QC decision
			appr = self.drop_rejections(df)
			rej = self.rejections_only(df)

			# Plot data subsets
			sns.histplot(appr[feature], kde=kde_appr, color='green', 
				binwidth=binwidth)
			sns.histplot(rej[feature], kde=kde_rej, color='red', 
				binwidth=binwidth)

			self.print_metadata(df, appr)

		else:
			self.print_metadata(df)
			sns.histplot(df[feature], kde=kde, binwidth=binwidth)

		self.print_test_case(df, feature, upper_bound, lower_bound)
		
		if show_bounds:
			plt.axvline(lower_bound, color='blue', linestyle='dashed', linewidth=2)
			plt.axvline(upper_bound, color='blue', linestyle='dashed', linewidth=2)

		plt.title(title)
		plt.xticks(rotation=45)
		plt.xlabel(xlabel) if xlabel else None
		plt.show()


	def bivarplot(self, df, x, y, title='Title', xlabel=None, ylabel=None, 
		upper_bound=.0000, lower_bound=.0000):
		"""A bivariate plot that allows for test cases.


		Parameters
		----------
		df : DataFrame
			The data source.
		x : str
			A DataFrame column name.
		y : str
			A DataFrame column name.
		title : str
			The plot title, optional. The default is 'Title'.
		xlabel : str or None
			The X axis label, optional. The default is None.
		ylabel : str or None
			The Y axis label, optional. The default is None.
		upper_bound : float
			The max tolerance for the test case, optional. The default is .0000.
		lower_bound : float
			The min tolerance for the test case, optional. The default is .0000.

		"""

		self.print_metadata(df, self.drop_rejections(df))
		self.print_test_case(df, x, upper_bound, lower_bound)
		sns.displot(data=df, x=x, y=y, cbar=True, height=Params.FIGURE_HEIGHT, aspect=1)
		plt.title(title)
		plt.xticks(rotation=45)
		plt.xlabel(xlabel) if xlabel else None
		plt.ylabel(ylabel) if ylabel else None
		plt.show()


	def scatter(self, df, x, y, title='Title', xlabel=None, ylabel=None, 
		test_var=None, upper_bound=.0000, lower_bound=.0000, representation=None, hue_target=None):
		"""A scatter plot that allows for test cases.


		Parameters
		----------
		df : DataFrame
			The data source.
		x : str
			A DataFrame column name.
		y : str
			A DataFrame column name.
		title : str
			The plot title, optional. The default is 'Title'.
		xlabel : str or None
			The X axis label, optional. The default is None.
		ylabel : str or None
			The Y axis label, optional. The default is None.
		test_var : str
			A DataFrame column name to run the test case against.
		upper_bound : float
			The max tolerance for the test case, optional. The default is .0000.
		lower_bound : float
			The min tolerance for the test case, optional. The default is .0000.
		representation : str or None
			Specifies the type of plot, optional. Valid parameters are:
			'decision' : Sets the color style per the QC review decision.
			'test' : Sets the color style per the specified test case.
			'auto_hue' : Automates the color style per the `hue_target`.
			None : Plots with no particular color style.
		hue_target : str or None
			The DataFrame column name upon which the hue is based.

		"""

		# Print high-level stats
		appr = self.drop_rejections(df)
		self.print_metadata(df, appr)

		if test_var:
			self.print_test_case(df, test_var, upper_bound, lower_bound)

		if representation == 'decision':
			rej = self.rejections_only(df)
			plt.scatter(x=appr[x].tolist(), y=appr[y].tolist(), color='green')
			plt.scatter(x=rej[x].tolist(), y=rej[y].tolist(), color='red')
		
		elif representation == 'test':
			# Strip instances of missing test variable
			df = df[~df[test_var].isna()]

			# Apply tolerance context
			inTol = df[(df[test_var] >= lower_bound) & (df[test_var] <= upper_bound)]
			outTol = df[(df[test_var] < lower_bound) | (df[test_var] > upper_bound)]

			# Plot according to test case
			plt.scatter(x=inTol[x].tolist(), y=inTol[y].tolist(), 
				color='green')
			plt.scatter(x=outTol[x].tolist(), y=outTol[y].tolist(), 
				color='red')
		
		elif representation == 'auto_hue':
			sns.relplot(data=df, x=x, y=y, hue=hue_target, palette='coolwarm',
			height=Params.FIGURE_HEIGHT, aspect=1)

		else:
			plt.scatter(x=df[x], y=df[y])

		plt.title(title)
		plt.xlabel(xlabel) if xlabel else None
		plt.ylabel(ylabel) if ylabel else None
		plt.show()	


	def plane_split(self, df):
		"""Divide a DataFrame into planar categories.
		
	
		Parameters
		----------
		df : DataFrame
			The data source.
	
		Returns
		-------
		A : DataFrame
			A DataFrame containing datum A features only.
		shoulders : DataFrame
			A DataFrame containing non-datum planes only.
		midplane : DataFrame
			A DataFrame containing alignment midplanes only.
			
		"""
		
		A = self.datum_only('A', df)
		shoulders = self.drop_datums(df)
		midplane = self.datum_only('midplane', df)  
		return A, shoulders, midplane


	def holes_only(self, df):
		"""Get a DataFrame subset that contains only the hole features.
		
	
		Parameters
		----------
		df : DataFrame
			The data source.
	
		Returns
		-------
		DataFrame
	
		"""

		return df[~pd.isna(df.position)]


	def drop_holes(self, df):
		"""Remove all holes features from a DataFrame.
		
	
		Parameters
		----------
		df : DataFrame
			The data source, optional. The default is None.
	
		Returns
		-------
		DataFrame
			A DataFrame subset without hole features.
	
		"""

		return df[pd.isna(df.position)]

		
	def cylinder_split(self, df):
		"""Divide a DataFrame into cylindrical categories.
		
	
		Parameters
		----------
		df : DataFrame
			The data source.
	
		Returns
		-------
		B : DataFrame
			A DataFrame containing datum B features only.
		C : DataFrame
			A DataFrame containing datum C features only.
		radials : DataFrame
			A DataFrame containing non-datum and non-hole features only.
		holes : DataFrame
			A DataFrame containing hole features only.
	
		"""
		
		B = self.datum_only('B', df)
		C = self.datum_only('C', df)
		radials = self.drop_datums(self.drop_holes(df))
		holes = self.drop_datums(self.holes_only(df))
		return B, C, radials, holes


	def get_B_with_wall(self, B_df, cylinders_df):
		"""Calculate and add the wall thickness to a DataFrame.


		Parameters
		----------
		B_df : DataFrame
			A DataFrame containing datum B features only.
		cylinders_df : DataFrame
			A DataFrame containing all cylindrical features.

		Returns
		-------
		B_with_wall : DataFrame
			A datum B DataFrame with an added 'wall_thickness' feature.

		Notes
		-----
		The wall thickness is calculated with the assumption that datum 
		cylinder B is the minimum diameter. This function was designed for seal 
		sleeves, which almost always satisfy this condition.

		"""

		B_with_wall = B_df.copy()

		for index, row in cylinders_df.iterrows():

			# Isolate individual part
			df = cylinders_df[cylinders_df.inspection_id == row.inspection_id]

			# Get min and max diameters
			max_dia = df.diameter_meas.max()
			bore_dia = df[df.feature == 'datum cylinder B'].diameter_meas.values[0]
			wall = max_dia - bore_dia

			# Update DataFrame with wall thickness
			B_with_wall.loc[B_with_wall.inspection_id == row.inspection_id, 'wall_thickness'] = wall
		
		return B_with_wall


	def get_B_with_intf_rate(self, B_df, rate_nominal=.0010):
		"""Calculate and add the interference rate to a DataFrame.


		Parameters
		----------
		B_df : DataFrame
			A DataFrame containing datum B features only.
		rate_nominal : float
			The desired rate of interference, optional. The default is .0010.

		Returns
		-------
		B_with_intf_rate : DataFrame
			A datum B DataFrame with an added 'intf_rate' feature.

		Notes
		-----
		The calculated interference rate is dependent on an assumed rate 
		nominal, and also an assumption that the datum cylinder B fits to the 
		shaft. This function was designed for seal sleeves, which almost always 
		satisfy this condition.

		"""

		Bpos = B_df[B_df.diameter_dev >= .0000]
		Bneg = B_df[B_df.diameter_dev < .0000]

		Bpos['intf_rate'] = (Bpos.diameter_nom * rate_nominal - Bpos.diameter_dev) / Bpos.diameter_nom

		Bneg['intf_rate'] = (Bneg.diameter_nom * rate_nominal + -1 * Bneg.diameter_dev) / Bneg.diameter_nom

		B_with_intf_rate = Bpos.append(Bneg)
		return B_with_intf_rate



