import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import warnings
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import Element
import tensorflow as tf
from scipy.ndimage import gaussian_filter
import argparse
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

substrates = ['SV', 'SV_1N', 'SV_2N', 'SV_3N',
			  'DV', 'DV_1N', 'DV_2N_1', 'DV_2N_2', 'DV_3N', 'DV_4N',
			  'HV', 'HV_1N', 'HV_2N_1', 'HV_2N_2', 'HV_3N', 'HV_4N']
# substrates = ['SV_3N']
elements = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
			'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
			'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
# elements = ['Pt']
meshs = ['42', '24', '22', '42_2']
# meshs = ['22']
add_Ns = ['0N', '1N', '2N', '3N']
# add_Ns = ['3N']
d_metals = {'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 5, 'Mn': 5, 'Fe': 6, 'Co': 7, 'Ni': 8, 'Cu': 10, 'Zn': 10,
			'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 5, 'Ru': 7, 'Rh': 8, 'Pd': 10, 'Ag': 10, 'Cd': 10,
			'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 6, 'Ir': 7, 'Pt': 9, 'Au': 10}


def atomic_distance(i, j):
	return (Element(i.specie).atomic_radius + Element(j.specie).atomic_radius) * 1.2


def expand_cell(struct, size, grid):
	desired_length = size[0] * grid + 1.0
	scaling_matrix = [1, 1, 1]
	for i, l in enumerate(struct.lattice.abc[:2]):
		# find scaling factor and make it even
		factor = desired_length // l + 1
		if factor % 2 == 0:
			factor += 1
		scaling_matrix[i] = np.int(factor)
	# print(scaling_matrix)
	struct.make_supercell(scaling_matrix=scaling_matrix)


def find_central_atom(struct):
	"""
	find central TM after cell expansion
	"""
	sites = []
	for site in struct.sites:
		if site.specie.symbol in elements:
			sites.append(site.coords)
	sites = np.array(sites)
	coords = np.zeros((3,))
	coords[0] = (sites[:, 0].max() + sites[:, 0].min()) / 2
	coords[1] = (sites[:, 1].max() + sites[:, 1].min()) / 2
	coords[2] = sites[:, 2][0]
	return coords


def find_min_z(struct):
	"""
	find lowest z coordination
	"""
	min_z = 100.
	for site in struct.sites:
		if site.z < min_z:
			min_z = site.z
	return min_z


def add_gaussian_blur(image, sigma=1.5):
	"""
	add gaussian blur to pixel image, image format: [channels, height, width]
	Args:
		image: pixel image
		sigma: greater sigma, more blurred
	Returns:
		pixel image after blurred
	"""
	for index, i in enumerate(image):
		image[index] = gaussian_filter(i, sigma=sigma)
	return image


def pixel(filepath, size=(6, 128, 128), grid=0.20, sigma=1.5, demo=True):
	"""
		create pixel image from POSCAR
	Args:
	    filepath: path to POSCAR
		size: [channel, width, height]
		grid: grid size
		sigma: gaussian blur
		demo: whether you use demo catalysts
	Returns:
		pixel image after blurred, [channel, height, width], numpy.ndarray
	"""
	if demo:
		struct = Poscar.from_file(os.path.join(filepath, 'POSCAR'),
		                          check_for_POTCAR=False, read_velocities=False).structure
	else:
		struct = Poscar.from_file(filepath,
		                          check_for_POTCAR=False, read_velocities=False).structure

	# expand cell
	expand_cell(struct, size[1:], grid)

	# find central metal atom coords in expanded cell and compress it to 2-dimensional
	central_atom_coords = find_central_atom(struct)[:2]
	image_length = size[1] * grid
	base_point = central_atom_coords - image_length / 2

	# initialize pixel image
	image = np.zeros(size)

	# find minimal z coordinate
	min_z = find_min_z(struct)

	for site in struct.sites:
		if site.x < base_point[0] or site.x >= base_point[0] + image_length:
			continue
		if site.y < base_point[1] or site.y >= base_point[1] + image_length:
			continue
		index_x = (site.x - base_point[0]) // grid
		index_y = (site.y - base_point[1]) // grid

		# image format: (channels, height, width)
		# set z coords as one channel
		image[0][np.int(index_x)][np.int(index_y)] = site.z - min_z
		# atomic number
		element = Element(site.specie)
		image[1][np.int(index_x)][np.int(index_y)] = element.Z
		# image[np.int(index_x)][np.int(index_y)][1] = 2 if element.Z == 6 else element.Z
		# electronegativity
		image[2][np.int(index_x)][np.int(index_y)] = element.X
		# row
		image[3][np.int(index_x)][np.int(index_y)] = element.row
		# group
		image[4][np.int(index_x)][np.int(index_y)] = element.group
		# atomic_radius_calculated
		image[5][np.int(index_x)][np.int(index_y)] = element.atomic_radius_calculated

	# Gaussian filter
	image = add_gaussian_blur(image, sigma=sigma)
	# print(np.shape(image))
	return image


def image_augment(image):
	"""
		data augmentation
	Args:
		image: pixel image, [channel, height, width]
	Returns:
		list of augmented images, [batch, channel, height, width]
	"""
	# translate image, create mini-cell by cropping supercell
	vectors = np.array([[16, 16], [12, 12], [8, 8], [4, 4],
						[12, 16], [8, 16], [4, 16],
						[16, 12], [16, 8], [16, 4]])
	vectors = vectors * 2

	# convert [channel, height, width] to [height, width, channel]
	image = np.transpose(image, (1, 2, 0))
	crop_images = np.array([tf.image.crop_to_bounding_box(image, i, j, 64, 64) for i, j in vectors])
	flip_images = tf.image.flip_left_right(crop_images).numpy()

	crop_images = tf.image.resize(images=crop_images, size=(32, 32), method='gaussian').numpy()
	flip_images = tf.image.resize(images=flip_images, size=(32, 32), method='gaussian').numpy()
	images = np.concatenate([crop_images, flip_images], axis=0)

	# [height, width, channel] to [channel, height, width]
	images = np.array([np.transpose(i, (2, 0, 1)) for i in images])
	return images


def save_image(image, data_path, filename):
	"""
	save one image as pickle file
	Args:
		image: pixel image, [channel, height, width]
		data_path: dir path that saves data
		filename: {id} or {id-aug}, i.e., 0 or 0-0
	"""
	with open(os.path.join(data_path, f"{filename}.pkl"), "wb") as f:
		pickle.dump(image, f)


def read_y(filepath):
	"""
	Returns:
		list of E_ads_OH
	"""
	df = pd.read_csv(os.path.join(filepath, "E_OH_all.csv"))
	return df["E_ads_H"].tolist()


def demo_pixels():
	"""
	create demo pixels
	"""
	# format: (channel, height, width)
	root_dir = os.getcwd()
	data_dir = os.path.join(root_dir, "data/pixel")
	catalysts_dir = os.path.join(root_dir, "demo_catalysts")

	outputs = read_y(root_dir)
	AUGMENT = True

	idx_image = 0  # index of image
	ids = []  # 0-0, 0-1, ..., 10-0, ...
	props = []

	for mesh in meshs:
		file_path_1 = os.path.join(catalysts_dir, mesh)
		if not os.path.exists(file_path_1):
			continue
		for add_N in add_Ns:
			file_path_2 = os.path.join(file_path_1, add_N)
			if not os.path.exists(file_path_2):
				continue
			for sub in substrates:
				file_path_3 = os.path.join(file_path_2, sub)
				if not os.path.exists(file_path_3):
					continue
				for e in elements:
					file_path_4 = os.path.join(file_path_3, e)
					if not os.path.exists(file_path_4):
						continue
					print(f"now processing: {file_path_4}")
					# create image from POSCAR
					image = pixel(file_path_4, demo=True)
					if AUGMENT:
						images = image_augment(image)
						idx_aug = 0  # augment index
						for i in images:
							idx_sample = f"{idx_image}-{idx_aug}"   # sample index
							ids.append(idx_sample)
							props.append(outputs[idx_image])
							save_image(i, data_dir, idx_sample)
							idx_aug += 1
					else:
						ids.append(idx_image)
						props.append(outputs[idx_image])
						save_image(image, data_dir, str(idx_image))
					idx_image += 1

	df = pd.DataFrame(np.array([ids, props]).T, columns=["id", "prop"])
	df.to_csv(os.path.join(root_dir, "data/id_prop.csv"), index=False)
	print("DONE")


def user_pixels():
	"""
	create user pixels
	"""
	# format: (channel, height, width)
	root_dir = os.getcwd()
	data_dir = os.path.join(root_dir, "data/pixel")
	catalysts_dir = os.path.join(root_dir, "user_catalysts")
	catalysts = [i for i in os.listdir(catalysts_dir) if "POSCAR" in i]

	outputs = read_y(root_dir)
	AUGMENT = True

	idx_image = 0  # index of image, total 2160
	ids = []  # 0-0, 0-1, ..., 10-0, ...
	props = []

	for cat in catalysts:
		print(f"now processing: {os.path.join(catalysts_dir, cat)}")
		# create image from POSCAR
		image = pixel(os.path.join(catalysts_dir, cat), demo=False)
		if AUGMENT:
			images = image_augment(image)
			idx_aug = 0  # augment index
			for i in images:
				idx_sample = f"{idx_image}-{idx_aug}"  # sample index
				ids.append(idx_sample)
				props.append(outputs[idx_image])
				save_image(i, data_dir, idx_sample)
				idx_aug += 1
		else:
			ids.append(idx_image)
			props.append(outputs[idx_image])
			save_image(image, data_dir, str(idx_image))
		idx_image += 1

	df = pd.DataFrame(np.array([ids, props]).T, columns=["id", "prop"])
	df.to_csv(os.path.join(root_dir, "data/id_prop.csv"), index=False)
	print("DONE")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--demo", action="store_true", help="whether use demo catalysts")
	args = parser.parse_args()

	if args.demo:
		demo_pixels()
	else:
		user_pixels()

