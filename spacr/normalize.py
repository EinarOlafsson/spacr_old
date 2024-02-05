import os
import gc
import time

import numpy as np
from skimage import filters, exposure

def normalize_stack(src, backgrounds=100, remove_background=False, lower_quantile=0.01, save_dtype=np.float32, signal_to_noise=5, signal_thresholds=1000, correct_illumination=False):
    if isinstance(signal_thresholds, int):
        signal_thresholds = [signal_thresholds]*4

    if isinstance(backgrounds, int):
        backgrounds = [backgrounds]*4

    paths = [os.path.join(src, file) for file in os.listdir(src) if file.endswith('.npz')]
    output_fldr = os.path.join(os.path.dirname(src), 'norm_channel_stack')
    os.makedirs(output_fldr, exist_ok=True)
    time_ls = []
    for file_index, path in enumerate(paths):
        with np.load(path) as data:
            stack = data['data']
            filenames = data['filenames']
        normalized_stack = np.zeros_like(stack, dtype=stack.dtype)
        file = os.path.basename(path)
        name, _ = os.path.splitext(file)
        
        for chan_index, channel in enumerate(range(stack.shape[-1])):
            single_channel = stack[:, :, :, channel]
            background = backgrounds[chan_index]
            signal_threshold = signal_thresholds[chan_index]
            if remove_background:
                single_channel[single_channel < background] = 0
            if correct_illumination:
                bg = filters.gaussian(single_channel, sigma=50)
                single_channel = single_channel - bg
            
            #Calculate the global lower and upper quantiles for non-zero pixels
            non_zero_single_channel = single_channel[single_channel != 0]
            global_lower = np.quantile(non_zero_single_channel, lower_quantile)
            for upper_p in np.linspace(0.98, 1.0, num=100).tolist():
                global_upper = np.quantile(non_zero_single_channel, upper_p)
                if global_upper >= signal_threshold:
                    break
            
            #Normalize the pixels in each image to the global quantiles and then dtype.
            arr_2d_normalized = np.zeros_like(single_channel, dtype=single_channel.dtype)
            signal_to_noise_ratio_ls = []
            for array_index in range(single_channel.shape[0]):
                start = time.time()
                arr_2d = single_channel[array_index, :, :]
                non_zero_arr_2d = arr_2d[arr_2d != 0]
                if non_zero_arr_2d.size > 0:
                    lower, upper = np.quantile(non_zero_arr_2d, (lower_quantile, upper_p))
                    signal_to_noise_ratio = upper/lower
                else:
                    signal_to_noise_ratio = 0
                signal_to_noise_ratio_ls.append(signal_to_noise_ratio)
                average_stnr = np.mean(signal_to_noise_ratio_ls) if len(signal_to_noise_ratio_ls) > 0 else 0
                if signal_to_noise_ratio > signal_to_noise:
                    arr_2d_rescaled = exposure.rescale_intensity(arr_2d, in_range=(lower, upper), out_range=(global_lower, global_upper))
                    arr_2d_normalized[array_index, :, :] = arr_2d_rescaled
                else:
                    arr_2d_normalized[array_index, :, :] = arr_2d
                stop = time.time()
                duration = (stop - start)*single_channel.shape[0]
                time_ls.append(duration)
                average_time = np.mean(time_ls) if len(time_ls) > 0 else 0
                print(f'Progress: files {file_index+1}/{len(paths)}, channels:{chan_index}/{stack.shape[-1]-1}, arrays:{array_index+1}/{single_channel.shape[0]}, Signal:{upper:.1f}, noise:{lower:.1f}, Signal-to-noise:{average_stnr:.1f}, Time/channel:{average_time:.2f}sec', end='\r', flush=True)
            normalized_single_channel = exposure.rescale_intensity(arr_2d_normalized, out_range='dtype')
            normalized_stack[:, :, :, channel] = normalized_single_channel
        save_loc = os.path.join(output_fldr, name+'_norm_stack.npz')
        normalized_stack = normalized_stack.astype(save_dtype)
        np.savez(save_loc, data=normalized_stack, filenames=filenames)
        del normalized_stack, single_channel, normalized_single_channel, stack, filenames
        gc.collect()
    return print(f'Saved stacks:{output_fldr}')