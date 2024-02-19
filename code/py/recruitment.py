import os
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries
from skimage.transform import rescale_intensity
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity
import math

def read_and_merge_data(locs, tables, verbose=False, include_multinucleated=False, include_multiinfected=False, include_noninfected=False):

    def read_db(db_loc, tables):
        conn = sqlite3.connect(db_loc)
        dfs = []
        for table in tables:
            query = f'SELECT * FROM {table}'
            df = pd.read_sql_query(query, conn)
            dfs.append(df)
        conn.close()
        return dfs

    def _split_data(df, group_by, object_type):

        df['prcfo'] = df['prcf'] + '_' + df[object_type]
        #df = df.drop([object_type], axis=1)
        # Set 'prcfo' as the index for both dataframes
        df = df.set_index(group_by, inplace=False)

        # Split the dataframe into numeric and non-numeric parts
        df_numeric = df.select_dtypes(include=np.number)
        df_non_numeric = df.select_dtypes(exclude=np.number)

        # Group by index (note that the result will be a GroupBy object)
        grouped_numeric = df_numeric.groupby(df_numeric.index).mean()
        grouped_non_numeric = df_non_numeric.groupby(df_non_numeric.index).first()
        return pd.DataFrame(grouped_numeric), pd.DataFrame(grouped_non_numeric)

    #Extract plate DataFrames
    all_dfs = []
    for loc in locs:
        db_dfs = read_db(loc, tables)
        all_dfs.append(db_dfs)
    
    #Extract Tables from DataFrames and concatinate rows
    for i, dfs in enumerate(all_dfs):
        if 'cell' in tables:
            cell = dfs[0]
            print(f'plate: {i+1} cells:{len(cell)}')

        if 'nucleus' in tables:
            nucleus = dfs[1]
            print(f'plate: {i+1} nuclei:{len(nucleus)} ')

        if 'pathogen' in tables:
            pathogen = dfs[2]
            
            print(f'plate: {i+1} pathogens:{len(pathogen)}')
        if 'cytoplasm' in tables:
            if not 'pathogen' in tables:
                cytoplasm = dfs[2]
            else:
                cytoplasm = dfs[3]
            print(f'plate: {i+1} cytoplasms: {len(cytoplasm)}')

        if i > 0:
            if 'cell' in tables:
                cells = pd.concat([cells, cell], axis = 0)
            if 'nucleus' in tables:
                nuclei = pd.concat([nuclei, nucleus], axis = 0)
            if 'pathogen' in tables:
                pathogens = pd.concat([pathogens, pathogen], axis = 0)
            if 'cytoplasm' in tables:
                cytoplasms = pd.concat([cytoplasms, cytoplasm], axis = 0)
        else:
            if 'cell' in tables:
                cells = cell.copy()
            if 'nucleus' in tables:
                nuclei = nucleus.copy()
            if 'pathogen' in tables:
                pathogens = pathogen.copy()
            if 'cytoplasm' in tables:
                cytoplasms = cytoplasm.copy()
    
    #Add an o in front of all object and cell lables to convert them to strings
    if 'cell' in tables:
        cells = cells.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cells = cells.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cells_g_df, metadata = _split_data(cells, 'prcfo', 'object_label')
        print(f'cells: {len(cells)}')
        print(f'cells grouped: {len(cells_g_df)}')
    if 'cytoplasm' in tables:
        cytoplasms = cytoplasms.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        cytoplasms = cytoplasms.assign(prcfo = lambda x: x['prcf'] + '_' + x['object_label'])
        cytoplasms_g_df, _ = _split_data(cytoplasms, 'prcfo', 'object_label')
        merged_df = cells_g_df.merge(cytoplasms_g_df, left_index=True, right_index=True)
        print(f'cytoplasms: {len(cytoplasms)}')
        print(f'cytoplasms grouped: {len(cytoplasms_g_df)}')
    if 'nucleus' in tables:
        nuclei = nuclei.dropna(subset=['cell_id'])
        nuclei = nuclei.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        nuclei = nuclei.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        nuclei = nuclei.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        nuclei['nuclei_prcfo_count'] = nuclei.groupby('prcfo')['prcfo'].transform('count')
        if include_multinucleated == False:
            #nuclei = nuclei[~nuclei['prcfo'].duplicated()]
            nuclei = nuclei[nuclei['nuclei_prcfo_count']==1]
        nuclei_g_df, _ = _split_data(nuclei, 'prcfo', 'cell_id')
        print(f'nuclei: {len(nuclei)}')
        print(f'nuclei grouped: {len(nuclei_g_df)}')
        if 'cytoplasm' in tables:
            merged_df = merged_df.merge(nuclei_g_df, left_index=True, right_index=True)
        else:
            merged_df = cells_g_df.merge(nuclei_g_df, left_index=True, right_index=True)
    if 'pathogen' in tables:
        pathogens = pathogens.dropna(subset=['cell_id'])
        pathogens = pathogens.assign(object_label=lambda x: 'o' + x['object_label'].astype(int).astype(str))
        pathogens = pathogens.assign(cell_id=lambda x: 'o' + x['cell_id'].astype(int).astype(str))
        pathogens = pathogens.assign(prcfo = lambda x: x['prcf'] + '_' + x['cell_id'])
        pathogens['pathogen_prcfo_count'] = pathogens.groupby('prcfo')['prcfo'].transform('count')
        if include_noninfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']>=1]
        if include_multiinfected == False:
            pathogens = pathogens[pathogens['pathogen_prcfo_count']<=1]
        pathogens_g_df, _ = _split_data(pathogens, 'prcfo', 'cell_id')
        print(f'pathogens: {len(pathogens)}')
        print(f'pathogens grouped: {len(pathogens_g_df)}')
        merged_df = merged_df.merge(pathogens_g_df, left_index=True, right_index=True)
    
    #Add prc column (plate row column)
    metadata = metadata.assign(prc = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col'])

    #Count cells per well
    cells_well = pd.DataFrame(metadata.groupby('prc')['object_label'].nunique())

    cells_well.reset_index(inplace=True)
    cells_well.rename(columns={'object_label': 'cells_per_well'}, inplace=True)
    metadata = pd.merge(metadata, cells_well, on='prc', how='inner', suffixes=('', '_drop_col'))
    object_label_cols = [col for col in metadata.columns if '_drop_col' in col]
    metadata.drop(columns=object_label_cols, inplace=True)

    #Add prcfo column (plate row column field object)
    metadata = metadata.assign(prcfo = lambda x: x['plate'] + '_' + x['row'] + '_' +x['col']+ '_' +x['field']+ '_' +x['object_label'])
    metadata.set_index('prcfo', inplace=True)

    merged_df = metadata.merge(merged_df, left_index=True, right_index=True)
    
    merged_df = merged_df.dropna(axis=1)
    
    print(f'Generated dataframe with: {len(merged_df.columns)} columns and {len(merged_df)} rows')
    
    obj_df_ls = []
    if 'cell' in tables:
        obj_df_ls.append(cells)
    if 'cytoplasm' in tables:
        obj_df_ls.append(cytoplasms)
    if 'nucleus' in tables:
        obj_df_ls.append(nuclei)
    if 'pathogen' in tables:
        obj_df_ls.append(pathogens)
        
    return merged_df, obj_df_ls

def annotate_conditions(df, cells=['HeLa'], cell_loc=None, pathogens=['rh'], pathogen_loc=None, treatments=['cm'], treatment_loc=None, types = ['col','col','col']):

    # Function to apply to each row
    def _map_values(row, dict_, type_='col'):
        for values, cols in dict_.items():
            if row[type_] in cols:
                return values
        return None

    if cell_loc is None:
        df['host_cells'] = cells[0]
    else:
        cells_dict = dict(zip(cells, cell_loc))
        df['host_cells'] = df.apply(lambda row: _map_values(row, cells_dict, type_=types[0]), axis=1)
    if pathogen_loc is None:
        if pathogens != None:
            df['pathogen'] = 'none'
    else:
        pathogens_dict = dict(zip(pathogens, pathogen_loc))
        df['pathogen'] = df.apply(lambda row: _map_values(row, pathogens_dict, type_=types[1]), axis=1)
    if treatment_loc is None:
        df['treatment'] = 'cm'
    else:
        treatments_dict = dict(zip(treatments, treatment_loc))
        df['treatment'] = df.apply(lambda row: _map_values(row, treatments_dict, type_=types[2]), axis=1)
    if pathogens != None:
        df['condition'] = df['pathogen']+'_'+df['treatment']
    else:
        df['condition'] = df['treatment']
    
    return df

def analyze_recruitment(src, metadata_settings, advanced_settings):

    def _group_by_well(df):
        numeric_cols = df._get_numeric_data().columns
        non_numeric_cols = df.select_dtypes(include=['object']).columns

        # Apply mean function to numeric columns and first to non-numeric
        df_grouped = df.groupby(['plate', 'row', 'col']).agg({**{col: np.mean for col in numeric_cols}, **{col: 'first' for col in non_numeric_cols}})
        return df_grouped

    def _plot_controls(df, mask_chans, channel_of_interest, figuresize=5):
        mask_chans.append(channel_of_interest)
        if len(mask_chans) == 4:
            mask_chans = [0,1,2,3]
        if len(mask_chans) == 3:
            mask_chans = [0,1,2]
        if len(mask_chans) == 2:
            mask_chans = [0,1]
        if len(mask_chans) == 1:
            mask_chans = [0]
        controls_cols = []
        for chan in mask_chans:

            controls_cols_c = []
            controls_cols_c.append(f'cell_channel_{chan}_mean_intensity')
            controls_cols_c.append(f'nucleus_channel_{chan}_mean_intensity')
            controls_cols_c.append(f'pathogen_channel_{chan}_mean_intensity')
            controls_cols_c.append(f'cytoplasm_channel_{chan}_mean_intensity')
            controls_cols.append(controls_cols_c)

        unique_conditions = df['condition'].unique().tolist()

        if len(unique_conditions) ==1:
            unique_conditions=unique_conditions+unique_conditions

        fig, axes = plt.subplots(len(unique_conditions), len(mask_chans)+1, figsize=(figuresize*len(mask_chans), figuresize*len(unique_conditions)))

        # Define RGB color tuples (scaled to 0-1 range)
        color_list = [(55/255, 155/255, 155/255), 
                      (155/255, 55/255, 155/255), 
                      (55/255, 155/255, 255/255), 
                      (255/255, 55/255, 155/255)]

        for idx_condition, condition in enumerate(unique_conditions):
            df_temp = df[df['condition'] == condition]
            for idx_channel, control_cols_c in enumerate(controls_cols):
                data = []
                std_dev = []
                for control_col in control_cols_c:
                    if control_col in df_temp.columns:
                        mean_intensity = df_temp[control_col].mean()
                        mean_intensity = 0 if np.isnan(mean_intensity) else mean_intensity
                        data.append(mean_intensity)
                        std_dev.append(df_temp[control_col].std())

                current_axis = axes[idx_condition][idx_channel]
                current_axis.bar(["cell", "nucleus", "pathogen", "cytoplasm"], data, yerr=std_dev, 
                                 capsize=4, color=color_list)
                current_axis.set_xlabel('Component')
                current_axis.set_ylabel('Mean Intensity')
                current_axis.set_title(f'Condition: {condition} - Channel {idx_channel}')
        plt.tight_layout()
        plt.show()

    def _plot_recruitment(df, df_type, channel_of_interest, target, columns=[], figuresize=50):

        color_list = [(55/255, 155/255, 155/255), 
                      (155/255, 55/255, 155/255), 
                      (55/255, 155/255, 255/255), 
                      (255/255, 55/255, 155/255)]

        sns.set_palette(sns.color_palette(color_list))
        font = figuresize/2
        width=figuresize
        height=figuresize/4

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(width, height))
        sns.barplot(ax=axes[0], data=df, x='condition', y=f'cell_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
        axes[0].set_xlabel(f'pathogen {df_type}', fontsize=font)
        axes[0].set_ylabel(f'cell_channel_{channel_of_interest}_mean_intensity', fontsize=font)

        sns.barplot(ax=axes[1], data=df, x='condition', y=f'nucleus_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
        axes[1].set_xlabel(f'pathogen {df_type}', fontsize=font)
        axes[1].set_ylabel(f'nucleus_channel_{channel_of_interest}_mean_intensity', fontsize=font)

        sns.barplot(ax=axes[2], data=df, x='condition', y=f'cytoplasm_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
        axes[2].set_xlabel(f'pathogen {df_type}', fontsize=font)
        axes[2].set_ylabel(f'cytoplasm_channel_{channel_of_interest}_mean_intensity', fontsize=font)

        sns.barplot(ax=axes[3], data=df, x='condition', y=f'pathogen_channel_{channel_of_interest}_mean_intensity', hue='pathogen', capsize=.1, ci='sd', dodge=False)
        axes[3].set_xlabel(f'pathogen {df_type}', fontsize=font)
        axes[3].set_ylabel(f'pathogen_channel_{channel_of_interest}_mean_intensity', fontsize=font)

        axes[0].legend_.remove()
        axes[1].legend_.remove()
        axes[2].legend_.remove()
        axes[3].legend_.remove()

        handles, labels = axes[3].get_legend_handles_labels()
        axes[3].legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
        for i in [0,1,2,3]:
            axes[i].tick_params(axis='both', which='major', labelsize=font)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

        columns = columns + ['pathogen_cytoplasm_mean_mean', 'pathogen_cytoplasm_q75_mean', 'pathogen_periphery_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_mean_mean', 'pathogen_outside_cytoplasm_q75_mean']
        columns = columns + [f'pathogen_slope_channel_{channel_of_interest}', f'pathogen_cell_distance_channel_{channel_of_interest}', f'nucleus_cell_distance_channel_{channel_of_interest}']

        width = figuresize*2
        columns_per_row = math.ceil(len(columns) / 2)
        height = (figuresize*2)/columns_per_row

        fig, axes = plt.subplots(nrows=2, ncols=columns_per_row, figsize=(width, height * 2))
        axes = axes.flatten()

        print(f'{columns}')

        for i, col in enumerate(columns):

            ax = axes[i]
            sns.barplot(ax=ax, data=df, x='condition', y=f'{col}', hue='pathogen', capsize=.1, ci='sd', dodge=False)
            ax.set_xlabel(f'pathogen {df_type}', fontsize=font)
            ax.set_ylabel(f'{col}', fontsize=int(font*2))
            ax.legend_.remove()
            ax.tick_params(axis='both', which='major', labelsize=font)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            if i <= 5:
                ax.set_ylim(1, None)

        for i in range(len(columns), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def _calculate_recruitment(df, channel):

        df['pathogen_cell_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'cell_channel_{channel}_mean_intensity']
        df['pathogen_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'cytoplasm_channel_{channel}_mean_intensity']
        df['pathogen_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_mean_intensity']/df[f'nucleus_channel_{channel}_mean_intensity']

        df['pathogen_cell_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'cell_channel_{channel}_mean_intensity']
        df['pathogen_cytoplasm_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'cytoplasm_channel_{channel}_mean_intensity']
        df['pathogen_nucleus_q75_mean'] = df[f'pathogen_channel_{channel}_percentile_75']/df[f'nucleus_channel_{channel}_mean_intensity']

        df['pathogen_outside_cell_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'cell_channel_{channel}_mean_intensity']
        df['pathogen_outside_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
        df['pathogen_outside_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_outside_mean']/df[f'nucleus_channel_{channel}_mean_intensity']

        df['pathogen_outside_cell_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'cell_channel_{channel}_mean_intensity']
        df['pathogen_outside_cytoplasm_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'cytoplasm_channel_{channel}_mean_intensity']
        df['pathogen_outside_nucleus_q75_mean'] = df[f'pathogen_channel_{channel}_outside_75_percentile']/df[f'nucleus_channel_{channel}_mean_intensity']

        df['pathogen_periphery_cell_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'cell_channel_{channel}_mean_intensity']
        df['pathogen_periphery_cytoplasm_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'cytoplasm_channel_{channel}_mean_intensity']
        df['pathogen_periphery_nucleus_mean_mean'] = df[f'pathogen_channel_{channel}_periphery_mean']/df[f'nucleus_channel_{channel}_mean_intensity']

        channels = [0,1,2,3]
        object_type = 'pathogen'
        for chan in channels:
            df[f'{object_type}_slope_channel_{chan}'] = 1

        object_type = 'nucleus'
        for chan in channels:
            df[f'{object_type}_slope_channel_{chan}'] = 1

        for chan in channels:
            df[f'nucleus_coordinates_{chan}'] = df[[f'nucleus_channel_{chan}_centroid_weighted_local-0', f'nucleus_channel_{chan}_centroid_weighted_local-1']].values.tolist()
            df[f'pathogen_coordinates_{chan}'] = df[[f'pathogen_channel_{chan}_centroid_weighted_local-0', f'pathogen_channel_{chan}_centroid_weighted_local-1']].values.tolist()
            df[f'cell_coordinates_{chan}'] = df[[f'cell_channel_{chan}_centroid_weighted_local-0', f'cell_channel_{chan}_centroid_weighted_local-1']].values.tolist()
            df[f'cytoplasm_coordinates_{chan}'] = df[[f'cytoplasm_channel_{chan}_centroid_weighted_local-0', f'cytoplasm_channel_{chan}_centroid_weighted_local-1']].values.tolist()

            df[f'pathogen_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'pathogen_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                          (row[f'pathogen_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
            df[f'nucleus_cell_distance_channel_{chan}'] = df.apply(lambda row: np.sqrt((row[f'nucleus_coordinates_{chan}'][0] - row[f'cell_coordinates_{chan}'][0])**2 + 
                                                          (row[f'nucleus_coordinates_{chan}'][1] - row[f'cell_coordinates_{chan}'][1])**2), axis=1)
        return df
    
    def _object_filter(df, object_type, size_range, intensity_range, mask_chans, mask_chan):
        if not size_range is None:
            if isinstance(size_range, list):
                if isinstance(size_range[0], int): 
                    df = df[df[f'{object_type}_area'] > size_range[0]]
                    print(f'After {object_type} minimum area filter: {len(df)}')
                if isinstance(size_range[1], int):
                    df = df[df[f'{object_type}_area'] < size_range[1]]
                    print(f'After {object_type} maximum area filter: {len(df)}')
        if not intensity_range is None:
            if isinstance(intensity_range, list):
                if isinstance(intensity_range[0], int):
                    df = df[df[f'{object_type}_channel_{mask_chans[mask_chan]}_mean_intensity'] > intensity_range[0]]
                    print(f'After {object_type} minimum mean intensity filter: {len(df)}')
                if isinstance(intensity_range[1], int):
                    df = df[df[f'{object_type}_channel_{mask_chans[mask_chan]}_mean_intensity'] < intensity_range[1]]
                    print(f'After {object_type} maximum mean intensity filter: {len(df)}')
        return df
    
    def _results_to_csv(src, df, df_well):
        cells = df
        wells = df_well
        results_loc = src+'/results'
        wells_loc = results_loc+'/wells.csv'
        cells_loc = results_loc+'/cells.csv'
        os.makedirs(results_loc, exist_ok=True)
        wells.to_csv(wells_loc, index=True, header=True)
        cells.to_csv(cells_loc, index=True, header=True)
        return cells, wells
    
    settings_dict = {**metadata_settings, **advanced_settings}
    settings_df = pd.DataFrame(list(settings_dict.items()), columns=['Key', 'Value'])
    settings_csv = os.path.join(src,'settings','analyze_settings.csv')
    os.makedirs(os.path.join(src,'settings'), exist_ok=True)
    settings_df.to_csv(settings_csv, index=False)

    # metadata settings
    target = metadata_settings['target']
    cell_types = metadata_settings['cell_types']
    cell_plate_metadata = metadata_settings['cell_plate_metadata']
    pathogen_types = metadata_settings['pathogen_types']
    pathogen_plate_metadata = metadata_settings['pathogen_plate_metadata']
    treatments = metadata_settings['treatments']
    treatment_plate_metadata = metadata_settings['treatment_plate_metadata']
    metadata_types = metadata_settings['metadata_types']
    channel_dims = metadata_settings['channel_dims']
    cell_chann_dim = metadata_settings['cell_chann_dim']
    cell_mask_dim = metadata_settings['cell_mask_dim']
    nucleus_chann_dim = metadata_settings['nucleus_chann_dim']
    nucleus_mask_dim = metadata_settings['nucleus_mask_dim']
    pathogen_chann_dim = metadata_settings['pathogen_chann_dim']
    pathogen_mask_dim = metadata_settings['pathogen_mask_dim']
    channel_of_interest = metadata_settings['channel_of_interest']
    
    # Advanced settings
    plot = advanced_settings['plot']
    plot_nr = advanced_settings['plot_nr']
    plot_control = advanced_settings['plot_control']
    figuresize = advanced_settings['figuresize']
    remove_background = advanced_settings['remove_background']
    backgrounds = advanced_settings['backgrounds']
    include_noninfected = advanced_settings['include_noninfected']
    include_multiinfected = advanced_settings['include_multiinfected']
    include_multinucleated = advanced_settings['include_multinucleated']
    cells_per_well = advanced_settings['cells_per_well']
    pathogen_size_range = advanced_settings['pathogen_size_range']
    nucleus_size_range = advanced_settings['nucleus_size_range']
    cell_size_range = advanced_settings['cell_size_range']
    pathogen_intensity_range = advanced_settings['pathogen_intensity_range']
    nucleus_intensity_range = advanced_settings['nucleus_intensity_range']
    cell_intensity_range = advanced_settings['cell_intensity_range']
    target_intensity_min = advanced_settings['target_intensity_min']
    
    print(f'Cell(s): {cell_types}, in {cell_plate_metadata}')
    print(f'Pathogen(s): {pathogen_types}, in {pathogen_plate_metadata}')
    print(f'Treatment(s): {treatments}, in {treatment_plate_metadata}')
    
    mask_dims=[cell_mask_dim,nucleus_mask_dim,pathogen_mask_dim]
    mask_chans=[nucleus_chann_dim, pathogen_chann_dim, cell_chann_dim]

    if isinstance(metadata_types, str):
        metadata_types = [metadata_types, metadata_types, metadata_types]
    if isinstance(metadata_types, list):
        if len(metadata_types) < 3:
            metadata_types = [metadata_types[0], metadata_types[0], metadata_types[0]]
            print(f'WARNING: setting metadata types to first element times 3: {metadata_types}. To avoid this behaviour, set metadata_types to a list with 3 elements. Elements should be col row or plate.')
        else:
            metadata_types = metadata_types
    
    if isinstance(backgrounds, (int,float)):
        backgrounds = [backgrounds, backgrounds, backgrounds, backgrounds]

    sns.color_palette("mako", as_cmap=True)
    print(f'channel:{channel_of_interest} = {target}')
    overlay_channels = [0, 1, 2, 3]
    overlay_channels.remove(channel_of_interest)
    overlay_channels.reverse()
    
    db_loc = [src+'/measurements/measurements.db']
    tables = ['cell', 'nucleus', 'pathogen','cytoplasm']
    df, object_dfs = read_and_merge_data(db_loc, 
                                         tables, 
                                         verbose=True, 
                                         include_multinucleated=include_multinucleated, 
                                         include_multiinfected=include_multiinfected, 
                                         include_noninfected=include_noninfected)
    
    df = annotate_conditions(df, 
                             cells=cell_types, 
                             cell_loc=cell_plate_metadata, 
                             pathogens=pathogen_types,
                             pathogen_loc=pathogen_plate_metadata,
                             treatments=treatments, 
                             treatment_loc=treatment_plate_metadata,
                             types=metadata_types)
    
    df = df.dropna(subset=['condition'])
    print(f'After dropping non-annotated wells: {len(df)} rows')
    files = df['file_name'].tolist()
    files = [item + '.npy' for item in files]
    random.shuffle(files)
    max_ = 100**10

    if plot:
        plot_settings = {'include_noninfected':include_noninfected, 
                         'include_multiinfected':include_multiinfected,
                         'include_multinucleated':include_multinucleated,
                         'remove_background':remove_background,
                         'filter_min_max':[[cell_size_min,max_],[nucleus_size_min,max_],[pathogen_size_min,max_],[0,max_]],
                         'channel_dims':channel_dims,
                         'backgrounds':backgrounds,
                         'cell_mask_dim':mask_dims[0],
                         'nucleus_mask_dim':mask_dims[1],
                         'pathogen_mask_dim':mask_dims[2],
                         'overlay_chans':[0,2,3],
                         'outline_thickness':3,
                         'outline_color':'gbr',
                         'overlay_chans':overlay_channels,
                         'overlay':True,
                         'normalization_percentiles':[1,99],
                         'normalize':True,
                         'print_object_number':True,
                         'nr':plot_nr,
                         'figuresize':20,
                         'cmap':'inferno',
                         'verbose':True}
        
    if os.path.exists(os.path.join(src,'merged')):
        plot_merged(src=os.path.join(src,'merged'), settings=plot_settings)
    
    if not cell_chann_dim is None:
        df = _object_filter(df, object_type='cell', size_range=cell_size_range, intensity_range=cell_intensity_range, mask_chans=mask_chans, mask_chan=0)
        if not target_intensity_min is None:
            df = df[df[f'cell_channel_{channel_of_interest}_percentile_95'] > target_intensity_min]
            print(f'After channel {channel_of_interest} filtration', len(df))
    if not nucleus_chann_dim is None:
        df = _object_filter(df, object_type='nucleus', size_range=nucleus_size_range, intensity_range=nucleus_intensity_range, mask_chans=mask_chans, mask_chan=1)
    if not pathogen_chann_dim is None:
        df = _object_filter(df, object_type='pathogen', size_range=pathogen_size_range, intensity_range=pathogen_intensity_range, mask_chans=mask_chans, mask_chan=2)
       
    df['recruitment'] = df[f'pathogen_channel_{channel_of_interest}_mean_intensity']/df[f'cytoplasm_channel_{channel_of_interest}_mean_intensity']
    for chan in channel_dims:
        df = _calculate_recruitment(df, channel=chan)
    print(f'calculated recruitment for: {len(df)} rows')
    df_well = _group_by_well(df)
    print(f'found: {len(df_well)} wells')
    
    df_well = df_well[df_well['cells_per_well'] >= cells_per_well]
    prc_list = df_well['prc'].unique().tolist()
    df = df[df['prc'].isin(prc_list)]
    print(f'After cells per well filter: {len(df)} cells in {len(df_well)} wells left wth threshold {cells_per_well}')
    
    if plot_control:
        _plot_controls(df, mask_chans, channel_of_interest, figuresize=5)

    print(f'PV level: {len(df)} rows')
    _plot_recruitment(df=df, df_type='by PV', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    print(f'well level: {len(df_well)} rows')
    _plot_recruitment(df=df_well, df_type='by well', channel_of_interest=channel_of_interest, target=target, figuresize=figuresize)
    cells,wells = _results_to_csv(src, df, df_well)
    return [cells,wells]
