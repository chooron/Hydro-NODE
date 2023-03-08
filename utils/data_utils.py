from datetime import datetime

import pandas as pd
import numpy as np
import os

from scipy.interpolate import PchipInterpolator

# data source path (download from https://ral.ucar.edu/solutions/products/camels)
base_data_path = r'E:\CAMELS\basin_dataset_public_v1p2'
source_dict = {'daymet': 'cida', 'maurer': 'maurer', 'nldas': 'nldas'}
forcing_header = ['Date', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']


def get_basin_huc(basin_id):
    """
    获取basin的流域分区信息
    :param basin_id: basin 的 id
    :return: 返回流域所属分区
    """
    if isinstance(basin_id, str):
        basin_id = int(basin_id)
    info_path = os.path.join(base_data_path, 'basin_metadata', 'gauge_info.xlsx')
    info_df = pd.read_excel(info_path, sheet_name='Sheet1')
    try:
        huc = info_df[info_df['GAGE_ID'] == basin_id]['HUC_02'].values[0]
    except:
        raise RuntimeError("未找到该basin id")
    return huc


def load_data(basin_id, forcing_source='daymet'):
    """
    获取流域数据
    :param basin_id: 流域id
    :return:
    """
    huc = str(get_basin_huc(basin_id)).zfill(2)
    if isinstance(basin_id, int):
        basin_id = str(basin_id).zfill(8)
    basin_forcing_file = os.path.join(base_data_path, 'basin_mean_forcing', forcing_source, huc,
                                      '{}_lump_{}_forcing_leap.txt'.format(basin_id, source_dict[forcing_source]))
    basin_flow_file = os.path.join(base_data_path, "usgs_streamflow", huc, "{}_streamflow_qc.txt".format(basin_id))

    with open(basin_forcing_file, 'r') as f:
        lines = f.readlines()
        area = int(lines[2].strip())
    forcing_df = pd.read_csv(basin_forcing_file, skiprows=4, sep='\t', header=None)
    forcing_df.columns = forcing_header
    forcing_df['Date'] = pd.to_datetime(forcing_df['Date'])
    raw_flow_df = pd.read_csv(basin_flow_file, header=None)
    flow_series = []
    for flow in raw_flow_df.values:
        flow_series.append(np.float32(flow[0].split(' ')[-2]))
    flow_series = np.array(flow_series)
    # handle data exceptions
    flow_series[np.where(flow_series == -999)] = np.nan
    # change data unit
    flow_series = flow_series * (304.8 ** 3) / (area * 10 ** 6) * 86400
    flow_df = pd.DataFrame(flow_series, columns=['flow'])
    flow_df['Date'] = forcing_df['Date']
    return forcing_df, flow_df


def prepare_data(forcing_df, flow_df):
    # split train and test dataset
    train_flow_df = flow_df[(flow_df['Date'] >= datetime(1980, 10, 1)) & (flow_df['Date'] <= datetime(2000, 9, 30))]
    test_flow_df = flow_df[(flow_df['Date'] >= datetime(2000, 10, 1)) & (flow_df['Date'] <= datetime(2010, 9, 30))]
    train_forcing_df = forcing_df[(forcing_df['Date'] >= datetime(1980, 10, 1))
                                  & (forcing_df['Date'] <= datetime(2000, 9, 30))]
    test_forcing_df = forcing_df[(forcing_df['Date'] >= datetime(2000, 10, 1))
                                 & (forcing_df['Date'] <= datetime(2010, 9, 30))]

    train_flow_df = train_flow_df.drop('Date', axis=1)
    test_flow_df = test_flow_df.drop('Date', axis=1)
    train_forcing_df = train_forcing_df.drop('Date', axis=1)
    test_forcing_df = test_forcing_df.drop('Date', axis=1)

    # extract the precipitation, temperature, and day length
    train_precp_series = train_forcing_df.loc[:, 'prcp(mm/day)'].values
    train_temp_series = train_forcing_df.loc[:, ['tmax(C)', 'tmin(C)']].values.mean(axis=1)
    train_Lday_series = train_forcing_df.loc[:, 'dayl(s)'].values / 3600

    test_precp_series = test_forcing_df.loc[:, 'prcp(mm/day)'].values
    test_temp_series = test_forcing_df.loc[:, ['tmax(C)', 'tmin(C)']].values.mean(axis=1)
    test_Lday_series = test_forcing_df.loc[:, 'dayl(s)'].values / 3600

    train_data_df = pd.DataFrame({'Precp': train_precp_series, 'Temp': train_temp_series, 'Lday': train_Lday_series})
    test_data_df = pd.DataFrame({'Precp': test_precp_series, 'Temp': test_temp_series, 'Lday': test_Lday_series})
    # interpolate the time series for solve ode
    all_data_df = pd.concat([train_data_df, test_data_df], join='inner')
    precp_series, temp_series, lday_series = all_data_df['Precp'].values, \
        all_data_df['Temp'].values, \
        all_data_df['Lday'].values
    t_series = np.linspace(0, len(precp_series) - 1, len(precp_series))
    precp_interp = PchipInterpolator(t_series, precp_series)
    temp_interp = PchipInterpolator(t_series, temp_series)
    lday_interp = PchipInterpolator(t_series, lday_series)
    return train_data_df, test_data_df,train_flow_df,test_flow_df, precp_interp, temp_interp, lday_interp


if __name__ == '__main__':
    forcing_df, flow_df = load_data(6431500)
