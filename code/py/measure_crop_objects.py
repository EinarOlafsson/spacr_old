## Standard Library Imports
import os
import re
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager

## Data Handling and Analysis
import numpy as np
import pandas as pd
import sqlite3

## Image Processing and Analysis
import cv2
from skimage.measure import regionprops, regionprops_table, label
from skimage.morphology import binary_dilation, binary_erosion, disk, remove_small_objects, square, dilation
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu, rank, gaussian
from skimage.color import label2rgb
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.exposure import rescale_intensity

## Scientific Computing
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, center_of_mass
from scipy.stats import entropy as shannon_entropy

## Visualization
import matplotlib.pyplot as plt

def __morphological_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, settings, zernike=True, degree=8):

    def get_components(cell_mask, nuclei_mask, pathogen_mask):
        # Create mappings from each cell to its nuclei, pathogens, and cytoplasms
        cell_to_nucleus = defaultdict(list)
        cell_to_pathogen = defaultdict(list)
        # Get unique cell labels
        cell_labels = np.unique(cell_mask)
        # Iterate over each cell label
        for cell_id in cell_labels:
            if cell_id == 0:
                continue
            # Find corresponding component labels
            nucleus_ids = np.unique(nuclei_mask[cell_mask == cell_id])
            pathogen_ids = np.unique(pathogen_mask[cell_mask == cell_id])
            # Update dictionaries, ignoring 0 (background) labels
            cell_to_nucleus[cell_id] = nucleus_ids[nucleus_ids != 0].tolist()
            cell_to_pathogen[cell_id] = pathogen_ids[pathogen_ids != 0].tolist()
        # Convert dictionaries to dataframes
        nucleus_df = pd.DataFrame(list(cell_to_nucleus.items()), columns=['cell_id', 'nucleus'])
        pathogen_df = pd.DataFrame(list(cell_to_pathogen.items()), columns=['cell_id', 'pathogen'])
        # Explode lists
        nucleus_df = nucleus_df.explode('nucleus')
        pathogen_df = pathogen_df.explode('pathogen')
        return nucleus_df, pathogen_df

    def _calculate_zernike(mask, df, degree=8):
        zernike_features = []
        for region in regionprops(mask):
            zernike_moment = zernike_moments(region.image, degree)
            zernike_features.append(zernike_moment.tolist())

        if zernike_features:
            feature_length = len(zernike_features[0])
            for feature in zernike_features:
                if len(feature) != feature_length:
                    raise ValueError("All Zernike moments must be of the same length")

            zernike_df = pd.DataFrame(zernike_features, columns=[f'zernike_{i}' for i in range(feature_length)])
            return pd.concat([df.reset_index(drop=True), zernike_df], axis=1)
        else:
            return df
        
    morphological_props = ['label', 'area', 'area_filled', 'area_bbox', 'convex_area', 'major_axis_length', 'minor_axis_length', 
                           'eccentricity', 'solidity', 'extent', 'perimeter', 'euler_number', 'equivalent_diameter_area', 'feret_diameter_max']
    
    prop_ls = []
    ls = []
    
    # Create mappings from each cell to its nuclei, pathogens, and cytoplasms
    if settings['cell_mask_dim'] is not None:
        cell_to_nucleus, cell_to_pathogen = get_components(cell_mask, nuclei_mask, pathogen_mask)
        cell_props = pd.DataFrame(regionprops_table(cell_mask, properties=morphological_props))
        cell_props = _calculate_zernike(cell_mask, cell_props, degree=degree)
        prop_ls = prop_ls + [cell_props]
        ls = ls + ['cell']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['cell']

    if settings['nuclei_mask_dim'] is not None:
        nucleus_props = pd.DataFrame(regionprops_table(nuclei_mask, properties=morphological_props))
        nucleus_props = _calculate_zernike(nuclei_mask, nucleus_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            nucleus_props = pd.merge(nucleus_props, cell_to_nucleus, left_on='label', right_on='nucleus', how='left')
        prop_ls = prop_ls + [nucleus_props]
        ls = ls + ['nucleus']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['nucleus']
    
    if settings['pathogen_mask_dim'] is not None:
        pathogen_props = pd.DataFrame(regionprops_table(pathogen_mask, properties=morphological_props))
        pathogen_props = _calculate_zernike(pathogen_mask, pathogen_props, degree=degree)
        if settings['cell_mask_dim'] is not None:
            pathogen_props = pd.merge(pathogen_props, cell_to_pathogen, left_on='label', right_on='pathogen', how='left')
        prop_ls = prop_ls + [pathogen_props]
        ls = ls + ['pathogen']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['pathogen']

    if settings['cytoplasm']:
        cytoplasm_props = pd.DataFrame(regionprops_table(cytoplasm_mask, properties=morphological_props))
        prop_ls = prop_ls + [cytoplasm_props]
        ls = ls + ['cytoplasm']
    else:
        prop_ls = prop_ls + [pd.DataFrame()]
        ls = ls + ['cytoplasm']

    df_ls = []
    for i,df in enumerate(prop_ls):
        df.columns = [f'{ls[i]}_{col}' for col in df.columns]
        df = df.rename(columns={col: 'label' for col in df.columns if 'label' in col})
        df_ls.append(df)
 
    return df_ls[0], df_ls[1], df_ls[2], df_ls[3]

def __intensity_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[3, 6, 12, 24], periphery=True, outside=True):
    
    def _create_dataframe(radial_distributions, object_type):
        df = pd.DataFrame()
        for key, value in radial_distributions.items():
            cell_label, object_label, channel_index = key
            for i in range(len(value)):
                col_name = f'{object_type}_rad_dist_channel_{channel_index}_bin_{i}'
                df.loc[object_label, col_name] = value[i]
            df.loc[object_label, 'cell_id'] = cell_label
        # Reset the index and rename the column that was previously the index
        df = df.reset_index().rename(columns={'index': 'label'})
        return df
    
    def _extended_regionprops_table(labels, image, intensity_props):
        regions = regionprops(labels, image)
        props = regionprops_table(labels, image, properties=intensity_props)
        percentiles = [5, 10, 25, 50, 75, 85, 95]
        for p in percentiles:
            props[f'percentile_{p}'] = [
                np.percentile(region.intensity_image.flatten()[~np.isnan(region.intensity_image.flatten())], p)
                for region in regions]
        return pd.DataFrame(props)

    def _calculate_homogeneity(label, channel, distances=[2,4,8,16,32,64]):
        homogeneity_values = []
        # Iterate through the regions in label_mask
        for region in regionprops(label):
            region_image = (region.image * channel[region.slice]).astype(int)
            homogeneity_per_distance = []
            for d in distances:
                rescaled_image = rescale_intensity(region_image, out_range=(0, 255)).astype('uint8')
                glcm = graycomatrix(rescaled_image, [d], [0], symmetric=True, normed=True)
                homogeneity_per_distance.append(graycoprops(glcm, 'homogeneity')[0, 0])
            homogeneity_values.append(homogeneity_per_distance)
        columns = [f'homogeneity_distance_{d}' for d in distances]
        homogeneity_df = pd.DataFrame(homogeneity_values, columns=columns)

        return homogeneity_df

    def _periphery_intensity(label_mask, image):
        periphery_intensity_stats = []
        boundary = find_boundaries(label_mask)
        for region in np.unique(label_mask)[1:]:  # skip the background label
            region_boundary = boundary & (label_mask == region)
            intensities = image[region_boundary]
            if intensities.size == 0:
                periphery_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            else:
                periphery_intensity_stats.append((region, np.mean(intensities), np.percentile(intensities,5), np.percentile(intensities,10),
                                                  np.percentile(intensities,25), np.percentile(intensities,50),
                                                  np.percentile(intensities,75), np.percentile(intensities,85), 
                                                  np.percentile(intensities,95)))
        return periphery_intensity_stats

    def _outside_intensity(label_mask, image, distance=5):
        outside_intensity_stats = []
        for region in np.unique(label_mask)[1:]:  # skip the background label
            region_mask = label_mask == region
            dilated_mask = binary_dilation(region_mask, iterations=distance)
            outside_mask = dilated_mask & ~region_mask
            intensities = image[outside_mask]
            if intensities.size == 0:
                outside_intensity_stats.append((region, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            else:
                outside_intensity_stats.append((region, np.mean(intensities), np.percentile(intensities,5), np.percentile(intensities,10),
                                                  np.percentile(intensities,25), np.percentile(intensities,50),
                                                  np.percentile(intensities,75), np.percentile(intensities,85), 
                                                  np.percentile(intensities,95)))
        return outside_intensity_stats
    
    def _calculate_radial_distribution(cell_mask, object_mask, channel_arrays, num_bins):
        
        def __calculate_average_intensity(distance_map, single_channel_image, num_bins):
            radial_distribution = np.zeros(num_bins)
            for i in range(num_bins):
                min_distance = i * (distance_map.max() / num_bins)
                max_distance = (i + 1) * (distance_map.max() / num_bins)
                bin_mask = (distance_map >= min_distance) & (distance_map < max_distance)
                radial_distribution[i] = single_channel_image[bin_mask].mean()
            return radial_distribution

        
        object_radial_distributions = {}

        # get unique cell labels
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0]

        for cell_label in cell_labels:
            cell_region = cell_mask == cell_label

            object_labels = np.unique(object_mask[cell_region])
            object_labels = object_labels[object_labels != 0]

            for object_label in nucleus_labels:
                objecyt_region = object_mask == object_label
                object_boundary = find_boundaries(objecyt_region, mode='outer')
                distance_map = distance_transform_edt(~object_boundary) * cell_region
                for channel_index in range(channel_arrays.shape[2]):
                    radial_distribution = __calculate_average_intensity(distance_map, channel_arrays[:, :, channel_index], num_bins)
                    object_radial_distributions[(cell_label, object_label, channel_index)] = radial_distribution

        return object_radial_distributions
    
    def _calculate_correlation_object_level(channel_image1, channel_image2, mask, settings):
        thresholds = settings['manders_thresholds']

        corr_data = {}
        for i in np.unique(mask)[1:]:
            object_mask = (mask == i)
            object_channel_image1 = channel_image1[object_mask]
            object_channel_image2 = channel_image2[object_mask]
            total_intensity1 = np.sum(object_channel_image1)
            total_intensity2 = np.sum(object_channel_image2)

            if len(object_channel_image1) < 2 or len(object_channel_image2) < 2:
                pearson_corr = np.nan
            else:
                pearson_corr, _ = pearsonr(object_channel_image1, object_channel_image2)

            corr_data[i] = {f'label_correlation': i,
                            f'Pearson_correlation': pearson_corr}

            for thresh in thresholds:
                chan1_thresh = np.percentile(object_channel_image1, thresh)
                chan2_thresh = np.percentile(object_channel_image2, thresh)

                # boolean mask where both signals are present
                overlap_mask = (object_channel_image1 > chan1_thresh) & (object_channel_image2 > chan2_thresh)
                M1 = np.sum(object_channel_image1[overlap_mask]) / total_intensity1 if total_intensity1 > 0 else 0
                M2 = np.sum(object_channel_image2[overlap_mask]) / total_intensity2 if total_intensity2 > 0 else 0

                corr_data[i].update({f'M1_correlation_{thresh}': M1,
                                     f'M2_correlation_{thresh}': M2})

        return pd.DataFrame(corr_data.values())
    
    def _estimate_blur(image):
        # Compute the Laplacian of the image
        lap = cv2.Laplacian(image, cv2.CV_64F)
        # Compute and return the variance of the Laplacian
        return lap.var()

    radial_dist = settings['radial_dist']
    calculate_correlation = settings['calculate_correlation']
    homogeneity = settings['homogeneity']
    distances = settings['homogeneity_distances']
    
    intensity_props = ["label", "centroid_weighted", "centroid_weighted_local", "max_intensity", "mean_intensity", "min_intensity"]
    col_lables = ['region_label', 'mean', '5_percentile', '10_percentile', '25_percentile', '50_percentile', '75_percentile', '85_percentile', '95_percentile']
    cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs = [], [], [], []
    ls = ['cell','nucleus','pathogen','cytoplasm']
    labels = [cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask]
    dfs = [cell_dfs, nucleus_dfs, pathogen_dfs, cytoplasm_dfs]
    
    for i in range(0,channel_arrays.shape[-1]):
        channel = channel_arrays[:, :, i]
        for j, (label, df) in enumerate(zip(labels, dfs)):
            
            if np.max(label) == 0:
                empty_df = pd.DataFrame()
                df.append(empty_df)
                continue
                
            mask_intensity_df = _extended_regionprops_table(label, channel, intensity_props) 
            mask_intensity_df['shannon_entropy'] = shannon_entropy(channel, base=2)

            if homogeneity:
                homogeneity_df = _calculate_homogeneity(label, channel, distances)
                mask_intensity_df = pd.concat([mask_intensity_df.reset_index(drop=True), homogeneity_df], axis=1)

            if periphery:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    periphery_intensity_stats = _periphery_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(periphery_intensity_stats, columns=[f'periphery_{stat}' for stat in col_lables])],axis=1)

            if outside:
                if ls[j] == 'nucleus' or ls[j] == 'pathogen':
                    outside_intensity_stats = _outside_intensity(label, channel)
                    mask_intensity_df = pd.concat([mask_intensity_df, pd.DataFrame(outside_intensity_stats, columns=[f'outside_{stat}' for stat in col_lables])], axis=1)

            blur_col = [_estimate_blur(channel[label == region_label]) for region_label in mask_intensity_df['label']]
            mask_intensity_df[f'{ls[j]}_channel_{i}_blur'] = blur_col

            mask_intensity_df.columns = [f'{ls[j]}_channel_{i}_{col}' if col != 'label' else col for col in mask_intensity_df.columns]
            df.append(mask_intensity_df)
    
    if radial_dist:
        if np.max(nuclei_mask) != 0:
            _calculate_radial_distribution(cell_mask, nuclei_mask, channel_arrays, num_bins)
            nucleus_df = _create_dataframe(nucleus_radial_distributions, 'nucleus')
            dfs[1].append(nucleus_df)
            
        if np.max(nuclei_mask) != 0:
            _calculate_radial_distribution(cell_mask, pathogen_mask, channel_arrays, num_bins)
            pathogen_df = _create_dataframe(pathogen_radial_distributions, 'pathogen')
            dfs[2].append(pathogen_df)
        
    if calculate_correlation:
        if channel_arrays.shape[-1] >= 2:
            for i in range(channel_arrays.shape[-1]):
                for j in range(i+1, channel_arrays.shape[-1]):
                    chan_i = channel_arrays[:, :, i]
                    chan_j = channel_arrays[:, :, j]
                    for m, mask in enumerate(labels):
                        coloc_df = _calculate_correlation_object_level(chan_i, chan_j, mask, settings)
                        coloc_df.columns = [f'{ls[m]}_channel_{i}_channel_{j}_{col}' for col in coloc_df.columns]
                        dfs[m].append(coloc_df)
    
    return pd.concat(cell_dfs, axis=1), pd.concat(nucleus_dfs, axis=1), pd.concat(pathogen_dfs, axis=1), pd.concat(cytoplasm_dfs, axis=1)

def _measure_crop_core(index, time_ls, file, settings):

    def ___get_percentiles(array, q1=2, q2=98):
        nimg = array.shape[2]
        percentiles = []
        for v in range(nimg):
            img = np.squeeze(array[:, :, v])
            non_zero_img = img[img > 0]
            if non_zero_img.size > 0: # check if there are non-zero values
                img_min = np.percentile(non_zero_img, q1)  # change percentile from 0.02 to 2
                img_max = np.percentile(non_zero_img, q2)  # change percentile from 0.98 to 98
                percentiles.append([img_min, img_max])
            else:  # if there are no non-zero values, just use the image as it is
                img_min, img_max = img.min(), img.max()
                percentiles.append([img_min, img_max])
        return percentiles

    def ___crop_center(img, cell_mask, new_width, new_height, normalize=(2,98)):
        # Convert all non-zero values in mask to 1
        cell_mask[cell_mask != 0] = 1
        mask_3d = np.repeat(cell_mask[:, :, np.newaxis], img.shape[2], axis=2).astype(img.dtype) # Create 3D mask
        img = np.multiply(img, mask_3d).astype(img.dtype) # Multiply image with mask to set pixel values outside of the mask to 0
        #centroid = np.round(ndi.measurements.center_of_mass(cell_mask)).astype(int) # Compute centroid of the mask
        centroid = np.round(ndi.center_of_mass(cell_mask)).astype(int) # Compute centroid of the mask
        # Pad the image and mask to ensure the crop will not go out of bounds
        pad_width = max(new_width, new_height)
        img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant')
        cell_mask = np.pad(cell_mask, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')
        # Update centroid coordinates due to padding
        centroid += pad_width
        # Compute bounding box
        start_y = max(0, centroid[0] - new_height // 2)
        end_y = min(start_y + new_height, img.shape[0])
        start_x = max(0, centroid[1] - new_width // 2)
        end_x = min(start_x + new_width, img.shape[1])
        # Crop to bounding box
        img = img[start_y:end_y, start_x:end_x, :]
        return img

    def ___plot_cropped_arrays(stack, figuresize=20,cmap='inferno'):
        """Plot arrays"""
        start = time.time()
        dim = stack.shape 
        channel=min(dim)
        if len(stack.shape) == 2:
            f, a = plt.subplots(1, 1,figsize=(figuresize,figuresize))
            a.imshow(stack, cmap=plt.get_cmap(cmap))
            a.set_title('Channel one',size=18)
            a.axis('off')
            f.tight_layout()
            plt.show()
        if len(stack.shape) > 2:
            anr = stack.shape[2]
            f, a = plt.subplots(1, anr,figsize=(figuresize,figuresize))
            for channel in range(anr):
                a[channel].imshow(stack[:,:,channel], cmap=plt.get_cmap(cmap))
                a[channel].set_title('Channel '+str(channel),size=18)
                a[channel].axis('off')
                f.tight_layout()
            plt.show()
        stop = time.time()
        duration = stop - start
        print('plot_cropped_arrays', duration)
        return

    def ___check_integrity(df):
        df.columns = [col + f'_{i}' if df.columns.tolist().count(col) > 1 and i != 0 else col for i, col in enumerate(df.columns)]
        label_cols = [col for col in df.columns if 'label' in col]
        df['label_list'] = df[label_cols].values.tolist()
        df['object_label'] = df['label_list'].apply(lambda x: x[0])
        df = df.drop(columns=label_cols)
        df['label_list'] = df['label_list'].astype(str)
        return df

    def ___filter_object(mask, min_value):
        count = np.bincount(mask.ravel())
        to_remove = np.where(count < min_value)
        mask[np.isin(mask, to_remove)] = 0
        return mask

    def ___safe_int_convert(value, default=0):
        try:
            return int(value)
        except ValueError:
            print(f'Could not convert {value} to int using {default}')
            return default

    def __map_wells(file_name, timelapse=False):
        try:
            parts = file_name.split('_')
            plate = 'p' + parts[0]
            well = parts[1]
            field = 'f' + str(___safe_int_convert(parts[2]))
            if timelapse:
                timeid = 't' + str(___safe_int_convert(parts[3]))
            if well[0].isalpha():
                row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
                column = 'c' + str(int(well[1:]))
            else:
                row, column = well, well
            if timelapse:    
                prcf = '_'.join([plate, row, column, field, timeid])
            else:
                prcf = '_'.join([plate, row, column, field])
        except Exception as e:
            print(f"Error processing filename: {file_name}")
            print(f"Error: {e}")
            plate, row, column, field, timeid, prcf = 'error','error','error','error','error', 'error'
        if timelapse:
            return plate, row, column, field, timeid, prcf
        else:
            return plate, row, column, field, prcf

    def __map_wells_png(file_name, timelapse=False):
        try:
            root, ext = os.path.splitext(file_name)
            parts = root.split('_')
            plate = 'p' + parts[0]
            well = parts[1]
            field = 'f' + str(___safe_int_convert(parts[2]))
            if timelapse:
                timeid = 't' + str(___safe_int_convert(parts[3]))
            object_id = 'o' + str(___safe_int_convert(parts[-1], default='none'))
            if well[0].isalpha():
                row = 'r' + str(string.ascii_uppercase.index(well[0]) + 1)
                column = 'c' + str(___safe_int_convert(well[1:]))
            else:
                row, column = well, well
            if timelapse:
                prcfo = '_'.join([plate, row, column, field, timeid, object_id])
            else:
                prcfo = '_'.join([plate, row, column, field, object_id])
        except Exception as e:
            print(f"Error processing filename: {file_name}")
            print(f"Error: {e}")
            plate, row, column, field, object_id, prcfo = 'error', 'error', 'error', 'error', 'error', 'error'
        if timelapse:
            return plate, row, column, field, timeid, prcfo, object_id,
        else:
            return plate, row, column, field, prcfo, object_id

    def __merge_and_save_to_database(morph_df, intensity_df, table_type, source_folder, file_name, experiment, timelapse=False):
        morph_df = ___check_integrity(morph_df)
        intensity_df = ___check_integrity(intensity_df)
        if len(morph_df) > 0 and len(intensity_df) > 0:
            merged_df = pd.merge(morph_df, intensity_df, on='object_label', how='outer')
            merged_df = merged_df.rename(columns={"label_list_x": "label_list_morphology", "label_list_y": "label_list_intensity"})
            merged_df['file_name'] = file_name
            merged_df['path_name'] = os.path.join(source_folder, file_name + '.npy')
            if timelapse:
                merged_df[['plate', 'row', 'col', 'field', 'timeid', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(__map_wells(x, timelapse)))
            else:
                merged_df[['plate', 'row', 'col', 'field', 'prcf']] = merged_df['file_name'].apply(lambda x: pd.Series(__map_wells(x, timelapse)))
            cols = merged_df.columns.tolist()  # get the list of all columns
            if table_type == 'cell' or table_type == 'cytoplasm':
                column_list = ['object_label', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
            elif table_type == 'nucleus' or table_type == 'pathogen':
                column_list = ['object_label', 'cell_id', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
            else:
                raise ValueError(f"Invalid table_type: {table_type}")
            # Check if all columns in column_list are in cols
            missing_columns = [col for col in column_list if col not in cols]
            if len(missing_columns) == 1 and missing_columns[0] == 'cell_id':
                missing_columns = False
                column_list = ['object_label', 'plate', 'row', 'col', 'field', 'prcf', 'file_name', 'path_name']
            if missing_columns:
                raise ValueError(f"Columns missing in DataFrame: {missing_columns}")
            for i, col in enumerate(column_list):
                cols.insert(i, cols.pop(cols.index(col)))
            merged_df = merged_df[cols]  # rearrange the columns
            if len(merged_df) > 0:
                try:
                    conn = sqlite3.connect(f'{source_folder}/measurements/measurements.db', timeout=5)
                    merged_df.to_sql(table_type, conn, if_exists='append', index=False)
                except sqlite3.OperationalError as e:
                    print("SQLite error:", e)

    def __exclude_objects(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, include_uninfected=True):
        # Remove cells with no nucleus or cytoplasm (or pathogen)
        filtered_cells = np.zeros_like(cell_mask) # Initialize a new mask to store the filtered cells.
        for cell_label in np.unique(cell_mask): # Iterate over all cell labels in the cell mask.
            if cell_label == 0: # Skip background
                continue
            cell_region = cell_mask == cell_label # Get a mask for the current cell.
            # Check existence of nucleus, cytoplasm and pathogen in the current cell.
            has_nucleus = np.any(nuclei_mask[cell_region])
            has_cytoplasm = np.any(cytoplasm_mask[cell_region])
            has_pathogen = np.any(pathogen_mask[cell_region])
            if include_uninfected:
                if has_nucleus and has_cytoplasm:
                    filtered_cells[cell_region] = cell_label
            else:
                if has_nucleus and has_cytoplasm and has_pathogen:
                    filtered_cells[cell_region] = cell_label
        # Remove objects outside of cells
        nuclei_mask = nuclei_mask * (filtered_cells > 0)
        pathogen_mask = pathogen_mask * (filtered_cells > 0)
        cytoplasm_mask = cytoplasm_mask * (filtered_cells > 0)
        return filtered_cells, nuclei_mask, pathogen_mask, cytoplasm_mask

    def __merge_overlapping_objects(mask1, mask2):
        labeled_1 = label(mask1)
        num_1 = np.max(labeled_1)
        for m1_id in range(1, num_1 + 1):
            current_1_mask = labeled_1 == m1_id
            overlapping_2_labels = np.unique(mask2[current_1_mask])
            overlapping_2_labels = overlapping_2_labels[overlapping_2_labels != 0]
            if len(overlapping_2_labels) > 1:
                overlap_percentages = [np.sum(current_1_mask & (mask2 == m2_label)) / np.sum(current_1_mask) * 100 for m2_label in overlapping_2_labels]
                max_overlap_label = overlapping_2_labels[np.argmax(overlap_percentages)]
                max_overlap_percentage = max(overlap_percentages)
                if max_overlap_percentage >= 90:
                    for m2_label in overlapping_2_labels:
                        if m2_label != max_overlap_label:
                            mask1[(current_1_mask) & (mask2 == m2_label)] = 0
                else:
                    for m2_label in overlapping_2_labels[1:]:
                        mask2[mask2 == m2_label] = overlapping_2_labels[0]
        return mask1, mask2

    def __generate_names(file_name, cell_id, cell_nuclei_ids, cell_pathogen_ids, source_folder, crop_mode='cell'):
        non_zero_cell_ids = cell_id[cell_id != 0]
        cell_id_str = "multi" if non_zero_cell_ids.size > 1 else str(non_zero_cell_ids[0]) if non_zero_cell_ids.size == 1 else "none"
        cell_nuclei_ids = cell_nuclei_ids[cell_nuclei_ids != 0]
        cell_nuclei_id_str = "multi" if cell_nuclei_ids.size > 1 else str(cell_nuclei_ids[0]) if cell_nuclei_ids.size == 1 else "none"
        cell_pathogen_ids = cell_pathogen_ids[cell_pathogen_ids != 0]
        cell_pathogen_id_str = "multi" if cell_pathogen_ids.size > 1 else str(cell_pathogen_ids[0]) if cell_pathogen_ids.size == 1 else "none"
        fldr = f"{source_folder}/data/"
        img_name = ""
        if crop_mode == 'nucleus':
            img_name = f"{file_name}_{cell_id_str}_{cell_nuclei_id_str}.png"
            fldr += "single_nucleus/" if cell_nuclei_ids.size == 1 else "multiple_nuclei/" if cell_nuclei_ids.size > 1 else "no_nucleus/"
            fldr += "single_pathogen/" if cell_pathogen_ids.size == 1 else "multiple_pathogens/" if cell_pathogen_ids.size > 1 else "uninfected/"
        elif crop_mode == 'pathogen':
            img_name = f"{file_name}_{cell_id_str}_{cell_pathogen_id_str}.png"
            fldr += "single_nucleus/" if cell_nuclei_ids.size == 1 else "multiple_nuclei/" if cell_nuclei_ids.size > 1 else "no_nucleus/"
            fldr += "infected/" if cell_pathogen_ids.size >= 1 else "uninfected/"
        elif crop_mode == 'cell' or crop_mode == 'cytoplasm':
            img_name = f"{file_name}_{cell_id_str}.png"
            fldr += "single_nucleus/" if cell_nuclei_ids.size == 1 else "multiple_nuclei/" if cell_nuclei_ids.size > 1 else "no_nucleus/"
            fldr += "single_pathogen/" if cell_pathogen_ids.size == 1 else "multiple_pathogens/" if cell_pathogen_ids.size > 1 else "uninfected/"
        parts = file_name.split('_')
        plate = parts[0]
        well = parts[1] 
        metadata = f'{plate}_{well}'
        fldr = os.path.join(fldr,metadata)
        table_name = fldr.replace("/", "_")
        return img_name, fldr, table_name

    def __find_bounding_box(crop_mask, _id, buffer=10):
        object_indices = np.where(crop_mask == _id)

        # Determine the bounding box coordinates
        y_min, y_max = object_indices[0].min(), object_indices[0].max()
        x_min, x_max = object_indices[1].min(), object_indices[1].max()

        # Add buffer to the bounding box coordinates
        y_min = max(y_min - buffer, 0)
        y_max = min(y_max + buffer, crop_mask.shape[0] - 1)
        x_min = max(x_min - buffer, 0)
        x_max = min(x_max + buffer, crop_mask.shape[1] - 1)

        # Create a new mask with the same dimensions as crop_mask
        new_mask = np.zeros_like(crop_mask)

        # Fill in the bounding box area with the _id
        new_mask[y_min:y_max+1, x_min:x_max+1] = _id

        return new_mask

    def __ssmovie(folder_paths):

        for folder_path in folder_paths:
            folder_path = os.path.join(folder_path, 'movies')
            os.makedirs(folder_path, exist_ok=True)
        
            # Regular expression to parse the filename
            filename_regex = re.compile(r'(\w+)_(\w+)_(\w+)_(\d+)_(\d+).png')

            # Dictionary to hold lists of images by plate, well, field, and object number
            grouped_images = defaultdict(list)

            # Iterate over all PNG files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    match = filename_regex.match(filename)
                    if match:
                        plate, well, field, time, object_number = match.groups()
                        key = (plate, well, field, object_number)
                        grouped_images[key].append((int(time), os.path.join(folder_path, filename)))

            for key, images in grouped_images.items():
                # Sort images by time using sorted and lambda function for custom sort key
                images = sorted(images, key=lambda x: x[0])
                _, image_paths = zip(*images)

                # Determine the size to which all images should be padded
                max_height = max_width = 0
                for image_path in image_paths:
                    image = cv2.imread(image_path)
                    h, w, _ = image.shape
                    max_height, max_width = max(max_height, h), max(max_width, w)

                # Initialize VideoWriter
                plate, well, field, object_number = key
                output_filename = f"{plate}_{well}_{field}_{object_number}.mp4"
                output_path = os.path.join(folder_path, output_filename)
                if not os.path.isfile(output_path):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(output_path, fourcc, 10, (max_width, max_height))

                    # Process each image
                    for image_path in image_paths:
                        image = cv2.imread(image_path)
                        h, w, _ = image.shape
                        padded_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                        padded_image[:h, :w, :] = image
                        video.write(padded_image)

                    video.release()
                    print(f"Movie saved to {output_path}")
                else:
                    continue

    start = time.time() 
    try:
        source_folder = os.path.dirname(settings['input_folder'])
        file_name = os.path.splitext(file)[0]
        data = np.load(os.path.join(settings['input_folder'], file))
        #print('data shape', data.shape)
        data_type = data.dtype
        if settings['save_measurements']:
            os.makedirs(source_folder+'/measurements', exist_ok=True)
            create_database(source_folder+'/measurements/measurements.db')    

        if settings['plot_filtration']:
            plot_cropped_arrays(data)

        channel_arrays = data[:, :, settings['channels']].astype(data_type)        
        if settings['cell_mask_dim'] is not None:
            cell_mask = data[:, :, settings['cell_mask_dim']].astype(data_type)
            if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
                cell_mask = ___filter_object(cell_mask, settings['cell_min_size'])
        else:
            cell_mask = np.zeros_like(data[:, :, 0])
            settings['cytoplasm'] = False
            settings['include_uninfected'] = True

        if settings['nuclei_mask_dim'] is not None:
            nuclei_mask = data[:, :, settings['nuclei_mask_dim']].astype(data_type)
            if settings['cell_mask_dim'] is not None:
                nuclei_mask, cell_mask = __merge_overlapping_objects(mask1=nuclei_mask, mask2=cell_mask)
            if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
                nuclei_mask = ___filter_object(nuclei_mask, settings['nucleus_min_size']) # Filter out small nuclei
        else:
            nuclei_mask = np.zeros_like(data[:, :, 0])

        if settings['pathogen_mask_dim'] is not None:
            pathogen_mask = data[:, :, settings['pathogen_mask_dim']].astype(data_type)
            if settings['merge_edge_pathogen_cells']:
                if settings['cell_mask_dim'] is not None:
                    pathogen_mask, cell_mask = __merge_overlapping_objects(mask1=pathogen_mask, mask2=cell_mask)
            if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
                pathogen_mask = ___filter_object(pathogen_mask, settings['pathogen_min_size']) # Filter out small pathogens
        else:
            pathogen_mask = np.zeros_like(data[:, :, 0])

        # Create cytoplasm mask
        if settings['cytoplasm']:
            if settings['cell_mask_dim'] is not None:
                if settings['nuclei_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
                    cytoplasm_mask = np.where(np.logical_or(nuclei_mask != 0, pathogen_mask != 0), 0, cell_mask)
                elif settings['nuclei_mask_dim'] is not None:
                    cytoplasm_mask = np.where(nuclei_mask != 0, 0, cell_mask)
                elif settings['pathogen_mask_dim'] is not None:
                    cytoplasm_mask = np.where(pathogen_mask != 0, 0, cell_mask)
                else:
                    cytoplasm_mask = np.zeros_like(cell_mask)
        else:
            cytoplasm_mask = np.zeros_like(cell_mask)

        if settings['cell_min_size'] is not None and settings['cell_min_size'] != 0:
            cell_mask = ___filter_object(cell_mask, settings['cell_min_size']) # Filter out small cells
        if settings['nucleus_min_size'] is not None and settings['nucleus_min_size'] != 0:
            nuclei_mask = ___filter_object(nuclei_mask, settings['nucleus_min_size']) # Filter out small nuclei
        if settings['pathogen_min_size'] is not None and settings['pathogen_min_size'] != 0:
            pathogen_mask = ___filter_object(pathogen_mask, settings['pathogen_min_size']) # Filter out small pathogens
        if settings['cytoplasm_min_size'] is not None and settings['cytoplasm_min_size'] != 0:
            cytoplasm_mask = ___filter_object(cytoplasm_mask, settings['cytoplasm_min_size']) # Filter out small cytoplasms

        if settings['cell_mask_dim'] is not None and settings['pathogen_mask_dim'] is not None:
            if settings['include_uninfected'] == False:
                cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask = __exclude_objects(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, include_uninfected=False)

        # Update data with the new masks
        if settings['cell_mask_dim'] is not None:
            data[:, :, settings['cell_mask_dim']] = cell_mask.astype(data_type)
        if settings['nuclei_mask_dim'] is not None:
            data[:, :, settings['nuclei_mask_dim']] = nuclei_mask.astype(data_type)
        if settings['pathogen_mask_dim'] is not None:
            data[:, :, settings['pathogen_mask_dim']] = pathogen_mask.astype(data_type)
        if settings['cytoplasm']:
            data = np.concatenate((data, cytoplasm_mask[:, :, np.newaxis]), axis=2)

        if settings['plot_filtration']:
            plot_cropped_arrays(data)

        if settings['save_measurements']:

            cell_df, nucleus_df, pathogen_df, cytoplasm_df = __morphological_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, settings)

            cell_intensity_df, nucleus_intensity_df, pathogen_intensity_df, cytoplasm_intensity_df = __intensity_measurements(cell_mask, nuclei_mask, pathogen_mask, cytoplasm_mask, channel_arrays, settings, sizes=[1, 2, 3, 4, 5], periphery=True, outside=True)
            if settings['cell_mask_dim'] is not None:
                cell_merged_df = __merge_and_save_to_database(cell_df, cell_intensity_df, 'cell', source_folder, file_name, settings['experiment'], settings['timelapse'])

            if settings['nuclei_mask_dim'] is not None:
                nucleus_merged_df = __merge_and_save_to_database(nucleus_df, nucleus_intensity_df, 'nucleus', source_folder, file_name, settings['experiment'], settings['timelapse'])

            if settings['pathogen_mask_dim'] is not None:
                pathogen_merged_df = __merge_and_save_to_database(pathogen_df, pathogen_intensity_df, 'pathogen', source_folder, file_name, settings['experiment'], settings['timelapse'])

            if settings['cytoplasm']:
                cytoplasm_merged_df = __merge_and_save_to_database(cytoplasm_df, cytoplasm_intensity_df, 'cytoplasm', source_folder, file_name, settings['experiment'], settings['timelapse'])

        ssmovie_folders = []
        if settings['save_png'] or settings['save_arrays'] or settings['plot']:

            if isinstance(settings['dialate_pngs'], bool):
                dialate_pngs = [settings['dialate_pngs'], settings['dialate_pngs'], settings['dialate_pngs']]
            if isinstance(settings['dialate_pngs'], list):
                dialate_pngs = settings['dialate_pngs']

            if isinstance(settings['dialate_png_ratios'], float):
                dialate_png_ratios = [settings['dialate_png_ratios'], settings['dialate_png_ratios'], settings['dialate_png_ratios']]

            if isinstance(settings['dialate_png_ratios'], list):
                dialate_png_ratios = settings['dialate_png_ratios']

            if isinstance(settings['crop_mode'], str):
                crop_mode = [settings['crop_mode']]
            if isinstance(settings['crop_mode'], list):
                crop_ls = settings['crop_mode']
                size_ls = settings['png_size']
                for crop_idx, crop_mode in enumerate(crop_ls):
                    width, height = size_ls[crop_idx]
                    if crop_mode == 'cell':
                        crop_mask = cell_mask.copy()
                        dialate_png = dialate_pngs[crop_idx]
                        dialate_png_ratio = dialate_png_ratios[crop_idx]
                    elif crop_mode == 'nucleus':
                        crop_mask = nuclei_mask.copy()
                        dialate_png = dialate_pngs[crop_idx]
                        dialate_png_ratio = dialate_png_ratios[crop_idx]
                    elif crop_mode == 'pathogen':
                        crop_mask = pathogen_mask.copy()
                        dialate_png = dialate_pngs[crop_idx]
                        dialate_png_ratio = dialate_png_ratios[crop_idx]
                    elif crop_mode == 'cytoplasm':
                        crop_mask = cytoplasm_mask.copy()
                        dialate_png = False
                    else:
                        print(f'Value error: Posseble values for crop_mode are: cell, nucleus, pathogen, cytoplasm')

                    objects_in_image = np.unique(crop_mask)
                    objects_in_image = objects_in_image[objects_in_image != 0]
                    img_paths = []

                    for _id in objects_in_image:
                        region = (crop_mask == _id)  # This creates a boolean mask for the region of interest

                        # Use the boolean mask to filter the cell_mask and then find unique IDs
                        region_cell_ids = np.atleast_1d(np.unique(cell_mask[region]))
                        region_nuclei_ids = np.atleast_1d(np.unique(nuclei_mask[region]))
                        region_pathogen_ids = np.atleast_1d(np.unique(pathogen_mask[region]))

                        if settings['use_bounding_box']:
                            region = __find_bounding_box(crop_mask, _id, buffer=10)

                        img_name, fldr, table_name = __generate_names(file_name=file_name, cell_id = region_cell_ids, cell_nuclei_ids=region_nuclei_ids, cell_pathogen_ids=region_pathogen_ids, source_folder=source_folder, crop_mode=crop_mode)

                        if dialate_png:
                            region_area = np.sum(region)
                            approximate_diameter = np.sqrt(region_area)
                            dialate_png_px = int(approximate_diameter * dialate_png_ratio) 
                            struct = generate_binary_structure(2, 2)
                            region = binary_dilation(region, structure=struct, iterations=dialate_png_px)

                        if settings['save_png']:
                            fldr_type = f"{crop_mode}_png/"
                            png_folder = os.path.join(fldr,fldr_type)
                            ssmovie_folders.append(png_folder)
                            img_path = os.path.join(png_folder, img_name)

                            png_channels = data[:, :, settings['png_dims']].astype(data_type)

                            if settings['normalize_by'] == 'fov':
                                percentiles_list = get_percentiles(png_channels, settings['normalize_percentiles'][0],q2=settings['normalize_percentiles'][1])

                            png_channels = ___crop_center(png_channels, region, new_width=width, new_height=height)

                            if isinstance(settings['normalize'], list):
                                if settings['normalize_by'] == 'png':
                                    png_channels = normalize_to_dtype(png_channels, q1=settings['normalize'][0],q2=settings['normalize'][1])
                                if settings['normalize_by'] == 'fov':
                                    png_channels = normalize_to_dtype(png_channels, q1=settings['normalize'][0],q2=settings['normalize'][1], percentiles=percentiles_list)

                            os.makedirs(png_folder, exist_ok=True)

                            if png_channels.shape[2] == 2:
                                dummy_channel = np.zeros_like(png_channels[:,:,0])  # Create a 2D zero array with same shape as one channel
                                png_channels = np.dstack((png_channels, dummy_channel))
                                cv2.imwrite(img_path, png_channels)
                            else:
                                cv2.imwrite(img_path, png_channels)

                            #if settings['save_measurements']:

                            img_paths.append(img_path)

                            if len(img_paths) == len(objects_in_image):

                                png_df = pd.DataFrame(img_paths, columns=['png_path'])

                                png_df['file_name'] = png_df['png_path'].apply(lambda x: os.path.basename(x))

                                parts = png_df['file_name'].apply(lambda x: pd.Series(__map_wells_png(x, timelapse=settings['timelapse'])))

                                columns = ['plate', 'row', 'col', 'field']

                                if settings['timelapse']:
                                    columns = columns + ['time_id']

                                columns = columns + ['prcfo']

                                if crop_mode == 'cell':
                                    columns = columns + ['cell_id']

                                if crop_mode == 'nucleus':
                                    columns = columns + ['nucleus_id']

                                if crop_mode == 'pathogen':
                                    columns = columns + ['pathogen_id']

                                if crop_mode == 'cytoplasm':
                                    columns = columns + ['cytoplasm_id']

                                png_df[columns] = parts

                                try:
                                    conn = sqlite3.connect(f'{source_folder}/measurements/measurements.db', timeout=5)
                                    png_df.to_sql('png_list', conn, if_exists='append', index=False)
                                    conn.commit()
                                except sqlite3.OperationalError as e:
                                    print(f"SQLite error: {e}", flush=True)

                            if settings['plot']:
                                plot_cropped_arrays(png_channels)

                        if settings['save_arrays']:
                            row_idx, col_idx = np.where(region)
                            region_array = data[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1, :]
                            array_folder = f"{fldr}/region_array/"            
                            os.makedirs(array_folder, exist_ok=True)
                            np.save(os.path.join(array_folder, img_name), region_array)
                            if plot:
                                plot_cropped_arrays(region_array)

                        if not settings['save_arrays'] and not settings['save_png'] and settings['plot']:
                            row_idx, col_idx = np.where(region)
                            region_array = data[row_idx.min():row_idx.max()+1, col_idx.min():col_idx.max()+1, :]
                            plot_cropped_arrays(region_array)

        if settings['timelapse']:
            if settings['save_png']:
                __ssmovie(ssmovie_folders)
                ssmovie_folders = []

        cells = np.unique(cell_mask)
    except Exception as e:
        print('main',e)
        cells = 0
        traceback.print_exc()

    end = time.time()
    duration = end-start
    time_ls.append(duration)
    average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
    return average_time, cells
    
def measure_crop(settings):

    def _save_settings_to_db(settings):
        # Convert the settings dictionary into a DataFrame
        settings_df = pd.DataFrame(list(settings.items()), columns=['setting_key', 'setting_value'])

        # Convert all values in the 'setting_value' column to strings
        settings_df['setting_value'] = settings_df['setting_value'].apply(str)
        display(settings_df)
        # Determine the directory path
        src = os.path.dirname(settings['input_folder'])
        directory = f'{src}/measurements'

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Database connection and saving the settings DataFrame
        conn = sqlite3.connect(f'{directory}/measurements.db', timeout=5)
        settings_df.to_sql('settings', conn, if_exists='replace', index=False)  # Replace the table if it already exists
        conn.close()

    #general settings
    settings['merge_edge_pathogen_cells'] = True
    settings['radial_dist'] = True
    settings['calculate_correlation'] = True
    settings['manders_thresholds'] = [15,85,95]
    settings['homogeneity'] = True
    settings['homogeneity_distances'] = [8,16,32]
    settings['save_arrays'] = False
    
    if settings['cell_mask_dim'] is None:
    	settings['include_uninfected'] = True
    
    if settings['cell_mask_dim'] is not None and settings['pathogen_min_size'] is not None:
    	settings['cytoplasm'] = True
    elif settings['cell_mask_dim'] is not None and settings['nucleus_min_size'] is not None:
    	settings['cytoplasm'] = True
    else:
    	settings['cytoplasm'] = False
    
    settings['center_crop'] = True

    int_setting_keys = ['cell_mask_dim', 'nuclei_mask_dim', 'pathogen_mask_dim', 'cell_min_size', 'nucleus_min_size', 'pathogen_min_size', 'cytoplasm_min_size']
    
    if isinstance(settings['normalize'], bool) and settings['normalize']:
        print(f'WARNING: to notmalize single object pngs set normalize to a list of 2 integers, e.g. [1,99] (lower and upper percentiles)')
        return

    if settings['normalize_by'] not in ['png', 'fov']:
        print("Warning: normalize_by should be either 'png' to notmalize each png to its own percentiles or 'fov' to normalize each png to the fov percentiles ")
        return

    if not all(isinstance(settings[key], int) or settings[key] is None for key in int_setting_keys):
        print(f"WARNING: {int_setting_keys} must all be integers")
        return

    if not isinstance(settings['channels'], list):
        print(f"WARNING: channels should be a list of integers representing channels e.g. [0,1,2,3]")
        return

    if not isinstance(settings['crop_mode'], list):
        print(f"WARNING: crop_mode should be a list with at least one element e.g. ['cell'] or ['cell','nucleus'] or [None]")
        return
    
    _save_settings_to_db(settings)

    files = [f for f in os.listdir(settings['input_folder']) if f.endswith('.npy')]
    max_workers = settings['max_workers'] or mp.cpu_count()-4
    print(f'using {max_workers} cpu cores')

    with mp.Manager() as manager:
        time_ls = manager.list()
        with mp.Pool(max_workers) as pool:
            result = pool.starmap_async(_measure_crop_core, [(index, time_ls, file, settings) for index, file in enumerate(files)])

            # Track progress in the main process
            while not result.ready():  # Run the loop until all tasks have finished
                time.sleep(1)  # Wait for a short amount of time to avoid excessive printing
                files_processed = len(time_ls)
                files_to_process = len(files)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                time_left = (((files_to_process-files_processed)*average_time)/max_workers)/60
                print(f'Progress: {files_processed}/{files_to_process} Time/img {average_time:.3f}sec, Time Remaining {time_left:.3f} min.', end='\r', flush=True)
            result.get()
    
