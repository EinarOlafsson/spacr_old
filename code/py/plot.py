## Data Manipulation
import numpy as np
import random

## Image Processing
from skimage.measure import find_contours
from skimage.morphology import dilation, square
from scipy import ndimage as ndi

##  Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl


def generate_mask_random_cmap(mask):  
    unique_labels = np.unique(mask)
    num_objects = len(unique_labels[unique_labels != 0])
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap
    
def random_cmap(num_objects=100):
    random_colors = np.random.rand(num_objects+1, 4)
    random_colors[:, 3] = 1
    random_colors[0, :] = [0, 0, 0, 1]
    random_cmap = mpl.colors.ListedColormap(random_colors)
    return random_cmap

def plot_arrays(src, figuresize=50, cmap='inferno', nr=1, normalize=True, q1=1, q2=99):
    mask_cmap = random_cmap()
    paths = []
    for file in os.listdir(src):
        if file.endswith('.npy'):
            path = os.path.join(src, file)
            paths.append(path)
    paths = random.sample(paths, nr)
    for path in paths:
        print(f'Image path:{path}')
        img = np.load(path)
        if normalize:
            img = normalize_to_dtype(array=img, q1=q1, q2=q2)
        dim = img.shape
        if len(img.shape)>2:
            array_nr = img.shape[2]
            fig, axs = plt.subplots(1, array_nr,figsize=(figuresize,figuresize))
            for channel in range(array_nr):
                i = np.take(img, [channel], axis=2)
                axs[channel].imshow(i, cmap=plt.get_cmap(cmap))
                axs[channel].set_title('Channel '+str(channel),size=24)
                axs[channel].axis('off')
        else:
            fig, ax = plt.subplots(1, 1,figsize=(figuresize,figuresize))
            ax.imshow(img, cmap=plt.get_cmap(cmap))
            ax.set_title('Channel 0',size=24)
            ax.axis('off')
        fig.tight_layout()
        plt.show()
    return

def plot_merged(src, settings):

    def __remove_noninfected(stack, cell_dim, nucleus_dim, pathogen_dim):
        if not cell_dim is None:
            cell_mask = stack[:, :, cell_dim]
        else:
            cell_mask = np.zeros_like(stack)
        if not nucleus_dim is None:
            nucleus_mask = stack[:, :, nucleus_dim]
        else:
            nucleus_mask = np.zeros_like(stack)

        if not pathogen_dim is None:
            pathogen_mask = stack[:, :, pathogen_dim]
        else:
            pathogen_mask = np.zeros_like(stack)

        for cell_label in np.unique(cell_mask)[1:]:
            cell_region = cell_mask == cell_label
            labels_in_cell = np.unique(pathogen_mask[cell_region])
            if len(labels_in_cell) <= 1:
                cell_mask[cell_region] = 0
                nucleus_mask[cell_region] = 0
        if not cell_dim is None:
            stack[:, :, cell_dim] = cell_mask
        if not nucleus_dim is None:
            stack[:, :, nucleus_dim] = nucleus_mask
        return stack

    def __remove_outside_objects(stack, cell_dim, nucleus_dim, pathogen_dim):
        if not cell_dim is None:
            cell_mask = stack[:, :, cell_dim]
        else:
            return stack
        nucleus_mask = stack[:, :, nucleus_dim]
        pathogen_mask = stack[:, :, pathogen_dim]
        pathogen_labels = np.unique(pathogen_mask)[1:]
        for pathogen_label in pathogen_labels:
            pathogen_region = pathogen_mask == pathogen_label
            cell_in_pathogen_region = np.unique(cell_mask[pathogen_region])
            cell_in_pathogen_region = cell_in_pathogen_region[cell_in_pathogen_region != 0]  # Exclude background
            if len(cell_in_pathogen_region) == 0:
                pathogen_mask[pathogen_region] = 0
                corresponding_nucleus_region = nucleus_mask == pathogen_label
                nucleus_mask[corresponding_nucleus_region] = 0
        stack[:, :, cell_dim] = cell_mask
        stack[:, :, nucleus_dim] = nucleus_mask
        stack[:, :, pathogen_dim] = pathogen_mask
        return stack

    def __remove_multiobject_cells(stack, mask_dim, cell_dim, nucleus_dim, pathogen_dim, object_dim):
        cell_mask = stack[:, :, mask_dim]
        nucleus_mask = stack[:, :, nucleus_dim]
        pathogen_mask = stack[:, :, pathogen_dim]
        object_mask = stack[:, :, object_dim]

        for cell_label in np.unique(cell_mask)[1:]:
            cell_region = cell_mask == cell_label
            labels_in_cell = np.unique(object_mask[cell_region])
            if len(labels_in_cell) > 2:
                cell_mask[cell_region] = 0
                nucleus_mask[cell_region] = 0
                for pathogen_label in labels_in_cell[1:]:  # Skip the first label (0)
                    pathogen_mask[pathogen_mask == pathogen_label] = 0

        stack[:, :, cell_dim] = cell_mask
        stack[:, :, nucleus_dim] = nucleus_mask
        stack[:, :, pathogen_dim] = pathogen_mask
        return stack
    
    def __generate_mask_random_cmap(mask):  
        unique_labels = np.unique(mask)
        num_objects = len(unique_labels[unique_labels != 0])
        random_colors = np.random.rand(num_objects+1, 4)
        random_colors[:, 3] = 1
        random_colors[0, :] = [0, 0, 0, 1]
        random_cmap = mpl.colors.ListedColormap(random_colors)
        return random_cmap
    
    def __get_colours_merged(outline_color):
        if outline_color == 'rgb':
            outline_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rgb
        elif outline_color == 'bgr':
            outline_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]  # bgr
        elif outline_color == 'gbr':
            outline_colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # gbr
        elif outline_color == 'rbg':
            outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
        else:
            outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]  # rbg
        return outline_colors
    
    def __filter_objects_in_plot(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, mask_dims, filter_min_max, include_multinucleated, include_multiinfected):

        stack = __remove_outside_objects(stack, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim)
        
        for i, mask_dim in enumerate(mask_dims):
            if not filter_min_max is None:
                min_max = filter_min_max[i]
            else:
                min_max = [0, 100000]

            mask = np.take(stack, mask_dim, axis=2)
            props = measure.regionprops_table(mask, properties=['label', 'area'])
            avg_size_before = np.mean(props['area'])
            total_count_before = len(props['label'])
            
            if not filter_min_max is None:
                valid_labels = props['label'][np.logical_and(props['area'] > min_max[0], props['area'] < min_max[1])]  
                stack[:, :, mask_dim] = np.isin(mask, valid_labels) * mask  

            props_after = measure.regionprops_table(stack[:, :, mask_dim], properties=['label', 'area']) 
            avg_size_after = np.mean(props_after['area'])
            total_count_after = len(props_after['label'])

            if mask_dim == cell_mask_dim:
                if include_multinucleated is False and nucleus_mask_dim is not None:
                    stack = __remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=pathogen_mask_dim)
                if include_multiinfected is False and cell_mask_dim is not None and pathogen_mask_dim is not None:
                    stack = __remove_multiobject_cells(stack, mask_dim, cell_mask_dim, nucleus_mask_dim, pathogen_mask_dim, object_dim=nucleus_mask_dim)
                cell_area_before = avg_size_before
                cell_count_before = total_count_before
                cell_area_after = avg_size_after
                cell_count_after = total_count_after
            if mask_dim == nucleus_mask_dim:
                nucleus_area_before = avg_size_before
                nucleus_count_before = total_count_before
                nucleus_area_after = avg_size_after
                nucleus_count_after = total_count_after
            if mask_dim == pathogen_mask_dim:
                pathogen_area_before = avg_size_before
                pathogen_count_before = total_count_before
                pathogen_area_after = avg_size_after
                pathogen_count_after = total_count_after

        if cell_mask_dim is not None:
            print(f'removed {cell_count_before-cell_count_after} cells, cell size from {cell_area_before} to {cell_area_after}')
        if nucleus_mask_dim is not None:
            print(f'removed {nucleus_count_before-nucleus_count_after} nuclei, nuclei size from {nucleus_area_before} to {nucleus_area_after}')
        if pathogen_mask_dim is not None:
            print(f'removed {pathogen_count_before-pathogen_count_after} pathogens, pathogen size from {pathogen_area_before} to {pathogen_area_after}')

        return stack
        
    def __normalize_and_outline(image, remove_background, backgrounds, normalize, normalization_percentiles, overlay, overlay_chans, mask_dims, outline_colors, outline_thickness):
        outlines = []
        if remove_background:
            for chan_index, channel in enumerate(range(image.shape[-1])):
                single_channel = stack[:, :, channel]  # Extract the specific channel
                background = backgrounds[chan_index]
                single_channel[single_channel < background] = 0
                image[:, :, channel] = single_channel
        if normalize:
            image = normalize_to_dtype(array=image, q1=normalization_percentiles[0], q2=normalization_percentiles[1])
        rgb_image = np.take(image, overlay_chans, axis=-1)
        rgb_image = rgb_image.astype(float)
        rgb_image -= rgb_image.min()
        rgb_image /= rgb_image.max()
        if overlay:
            overlayed_image = rgb_image.copy()
            for i, mask_dim in enumerate(mask_dims):
                mask = np.take(stack, mask_dim, axis=2)
                outline = np.zeros_like(mask)
                # Find the contours of the objects in the mask
                for j in np.unique(mask)[1:]:
                    contours = find_contours(mask == j, 0.5)
                    for contour in contours:
                        contour = contour.astype(int)
                        outline[contour[:, 0], contour[:, 1]] = j
                # Make the outline thicker
                outline = dilation(outline, square(outline_thickness))
                outlines.append(outline)
                # Overlay the outlines onto the RGB image
                for j in np.unique(outline)[1:]:
                    overlayed_image[outline == j] = outline_colors[i % len(outline_colors)]
            return overlayed_image, image, outlines
        else:
            return [], image, []
        
    def __plot_merged_plot(overlay, image, stack, mask_dims, figuresize, overlayed_image, outlines, cmap, outline_colors, print_object_number):
        if overlay:
            fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims) + 1, figsize=(4 * figuresize, figuresize))
            ax[0].imshow(overlayed_image)
            ax[0].set_title('Overlayed Image')
            ax_index = 1
        else:
            fig, ax = plt.subplots(1, image.shape[-1] + len(mask_dims), figsize=(4 * figuresize, figuresize))
            ax_index = 0

        # Normalize and plot each channel with outlines
        for v in range(0, image.shape[-1]):
            channel_image = image[..., v]
            channel_image_normalized = channel_image.astype(float)
            channel_image_normalized -= channel_image_normalized.min()
            channel_image_normalized /= channel_image_normalized.max()
            channel_image_rgb = np.dstack((channel_image_normalized, channel_image_normalized, channel_image_normalized))

            # Apply the outlines onto the RGB image
            for outline, color in zip(outlines, outline_colors):
                for j in np.unique(outline)[1:]:
                    channel_image_rgb[outline == j] = mpl.colors.to_rgb(color)

            ax[v + ax_index].imshow(channel_image_rgb)
            ax[v + ax_index].set_title('Image - Channel'+str(v))

        for i, mask_dim in enumerate(mask_dims):
            mask = np.take(stack, mask_dim, axis=2)
            random_cmap = __generate_mask_random_cmap(mask)
            ax[i + image.shape[-1] + ax_index].imshow(mask, cmap=random_cmap)
            ax[i + image.shape[-1] + ax_index].set_title('Mask '+ str(i))
            if print_object_number:
                unique_objects = np.unique(mask)[1:]
                for obj in unique_objects:
                    cy, cx = ndi.center_of_mass(mask == obj)
                    ax[i + image.shape[-1] + ax_index].text(cx, cy, str(obj), color='white', fontsize=font, ha='center', va='center')

        plt.tight_layout()
        plt.show()
        return fig
    
    font = settings['figuresize']/2
    outline_colors = __get_colours_merged(settings['outline_color'])
    index = 0
    
    mask_dims = [settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim']]
    mask_dims = [element for element in mask_dims if element is not None]
    
    if settings['verbose']:
        display(settings)

    for file in os.listdir(src):
        path = os.path.join(src, file)
        stack = np.load(path)
        print(f'Loaded: {path}')
        if not settings['include_noninfected']:
            if settings['pathogen_mask_dim'] is not None and settings['cell_mask_dim'] is not None:
                stack = __remove_noninfected(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'])

        if settings['include_multiinfected'] is not True or settings['include_multinucleated'] is not True or settings['filter_min_max'] is not None:
            stack = __filter_objects_in_plot(stack, settings['cell_mask_dim'], settings['nucleus_mask_dim'], settings['pathogen_mask_dim'], mask_dims, settings['filter_min_max'], settings['include_multinucleated'], settings['include_multiinfected'])

        image = np.take(stack, settings['channel_dims'], axis=2)

        overlayed_image, image, outlines = __normalize_and_outline(image, settings['remove_background'], settings['backgrounds'], settings['normalize'], settings['normalization_percentiles'], settings['overlay'], settings['overlay_chans'], mask_dims, outline_colors, settings['outline_thickness'])
        
        if index < settings['nr']:
            index += 1
            fig = __plot_merged_plot(settings['overlay'], image, stack, mask_dims, settings['figuresize'], overlayed_image, outlines, settings['cmap'], outline_colors, settings['print_object_number'])
        else:
            return
