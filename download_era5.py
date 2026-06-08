import cdsapi
import os 

if __name__ == "__main__":
        

    print("Starting download...")
    
    path="../data/temp/" #temporary path
    final_path = "../data/era5/" #save path

    years_per_download = 1
    start, end = 0, years_per_download

    #'1980', 
    years_to_download = ['1981', '1982', '1983', '1984', '1985', '1986', '1987','1988', '1989',
                        '1990','1991', '1992', '1993', '1994','1995', '1996', '1997', '1998','1999',
                        '2000', '2001','2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                        '2010','2011', '2012','2013', '2014', '2015', '2016', '2017', '2018', '2019',
                        '2020', '2021', '2022', '2023','2024', '2025']

    area = [5.3, -73.9, -33.9, -34.9]  #brazil area
    
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="b0474ce3-c4ed-4747-b666-66dcca14d8ea"
    )
    

    while start < len(years_to_download):

        cur_years = years_to_download[start:end]
        
        print(f"Downloading data for years {cur_years[0]} to {cur_years[-1]}...")

        filename = client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":[ 
                            # "2m_temperature",
                            # "total_precipitation",
                            # "surface_pressure",
                            # "2m_dewpoint_temperature",
                            "maximum_2m_temperature_since_previous_post_processing",
                            "minimum_2m_temperature_since_previous_post_processing"
                                ],
                "year": cur_years,
                "month": ["01", "02", "03","04", "05", "06",
                            "07", "08", "09","10", "11", "12"],
                "day":  [ "01", "02", "03","04", "05", "06",
                            "07", "08", "09","10", "11", "12",
                            "13", "14", "15","16", "17", "18",
                            "19", "20", "21","22", "23", "24",
                            "25", "26", "27","28", "29", "30","31"],
                "time": ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                         "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                         "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                         "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"],
                "data_format": "netcdf",
                "download_format": "zip",
                "area": area,
                }
        ).download()

        start = end
        end += years_per_download

        print(f"Download finished for years {cur_years[0]} to {cur_years[-1]}\n")

        os.system(f"unzip {filename} -d {path}")
        os.system(f"rm {filename}")
        new_files = [i for i in os.listdir(path) if i.endswith(".nc")]
        for file in new_files:     #rename files 
            # new_name = f"{min(years_to_download)}-{max(years_to_download)}1]-{file.split('-')[-}"
            new_name = f"{cur_years[0]}-min-max-temp-{file.split('-')[-1]}"
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            os.system(f"mv {os.path.join(path, new_name)} {os.path.join(final_path, new_name)}")
            
    