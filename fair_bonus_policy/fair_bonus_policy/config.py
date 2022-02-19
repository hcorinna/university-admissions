from pathlib import Path

data_dir = Path('/data') # specify the path for the data directory
career_data_path = data_dir / 'school_data_2016.csv'

additional_data_dir = Path('/additional_data') # specify the path for the additional data directory

image_dir = Path('/Pictures') # specify the path to the image folder

income_low = 2 # low-income: smaller or equal
income_bounds = [income_low]

q1 = 548
q2 = 616
q3 = 682
nem_bounds = [q1,q2,q3]