import click
import numpy as np
import datetime

def check_if_inside(csv_line, lat_window, lon_window, time_window):
    if np.count_nonzero(lat_window == -1000) == (lat_window.shape[0] * lat_window.shape[1]):
        return False
    csv_line = csv_line.strip().split(',')
    csv_year = int(csv_line[0][0:4])
    csv_month = int(csv_line[0][4:6])
    csv_day = int(csv_line[0][6:8])
    csv_time = csv_line[1].split(':')
    csv_hour = int(csv_time[0])
    csv_minute = int(csv_time[1])
    csv_second = int(csv_time[2])
    csv_datetime = datetime.datetime(csv_year, csv_month, csv_day, csv_hour, csv_minute, csv_second)
    csv_lat = float(csv_line[2])
    csv_lon = float(csv_line[3])
    mask = lat_window != -1000
    min_datetime = time_window[mask].min()
    max_datetime = time_window[mask].max()
    min_lat = lat_window[mask].min()
    max_lat = lat_window[mask].max()
    min_lon = lon_window[mask].min()
    max_lon = lon_window[mask].max()
    return ((min_datetime <= csv_datetime <= max_datetime) and
            (min_lat <= csv_lat <= max_lat) and
            (min_lon <= csv_lon <= max_lon))

@click.command()
@click.option('-i', '--input-file', help='Input CSV with super-emitter locations')
@click.option('-m', '--matrix-file', help='Input NPZ file with methane data from TROPOMI')
def main(input_file, matrix_file):
    methane_data = np.load(matrix_file, allow_pickle=True)
    with open(input_file, 'r') as fd:
        data = fd.readlines()[1:]
    methane_matrix = methane_data['xch4_corrected']
    lat_matrix = methane_data['lat']
    lon_matrix = methane_data['lon']
    time_matrix = methane_data['time']
    start_date = time_matrix.min()
    end_date = time_matrix.max()
    rows, cols = methane_matrix.shape
    print(f"Examining {matrix_file}!")
    for row in range(0, rows, 16):
        for col in range(0, cols, 16):
            print(f"Examining patch ({row}, {col})")
            if row + 32 < rows and col + 32 < cols:
                methane_window = methane_matrix[row:row + 32][:, col:col + 32]
                lat_window = lat_matrix[row:row + 32][:, col:col + 32]
                lon_window = lon_matrix[row:row + 32][:, col:col + 32]
                time_window = time_matrix[row:row + 32][:, col:col + 32]
                for csv_line in data:
                    if check_if_inside(csv_line, lat_window, lon_window, time_window):
                        print(f"FOUND! {csv_line} in {matrix_file}")
                    
if __name__ == '__main__':
    main()
