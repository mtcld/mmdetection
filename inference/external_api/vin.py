import hashlib
from urllib.parse import urljoin
import requests
import json
import csv
import pickle
class Car_Info():
    def __init__(self):
        make = "N/A"
        model = "N/A"
        year = "N/A"


    def get_car_info_from_vin(self, vin):
        apiPrefix = "https://api.vindecoder.eu/3.0/"
        apikey = "a9db72097e94_"
        secretkey = "d795b6f76a_"
        f_id = "decode"

        base = vin + "|"  + f_id  + "|" + apikey + "|" + secretkey
        controlsum =  hashlib.sha1(base.encode()).hexdigest()[:10]

        url = urljoin(apiPrefix, "/".join([apikey , controlsum, "decode", vin + ".json"]))
        # print(url)
        r = requests.get(url)
        if r.ok:
            if "decode" in r.json():
                return r.json()["decode"]
        return None

    def get_car_info_fake(self, vin):
        j = [{'label': 'Make', 'value': 'Skoda'}, {'label': 'Manufacturer', 'value': 'Skoda Auto AS'}, {'label': 'Plant Country', 'value': 'Czech Republic'}, {'label': 'Product Type', 'value': 'Passenger car'}, {'label': 'Manufacturer Address', 'value': 'ECO b.c.1, 293 60 Mlada Boleslav, Czech Republic'}, {'label': 'Check Digit', 'value': 'X'}, {'label': 'Sequential Number', 'value': '275210'}, {'label': 'Model', 'value': 'Octavia'}, {'label': 'Model Year', 'value': 2016}, {'label': 'Body', 'value': 'Wagon'}, {'label': 'Number of Seats', 'value': 5}, {'label': 'Number of Doors', 'value': 5}, {'label': 'Weight Empty (kg)', 'value': 1327}, {'label': 'Wheelbase (mm)', 'value': 2686}, {'label': 'Wheel Size', 'value': '195/65 R15 91H '}, {'label': 'Fuel Type - Primary', 'value': 'Diesel'}, {'label': 'Engine Displacement (ccm)', 'value': 1598}, {'label': 'Engine Power (kW)', 'value': 67}, {'label': 'Transmission', 'value': 'Manual/Standard'}, {'label': 'Series', 'value': 'ACCXXAX0 NFM5FM5A4*'}, {'label': 'Color', 'value': 'White'}, {'label': 'Registered', 'value': '2016-06-09'}, {'label': 'Engine Cylinders Position', 'value': 'Inline'}, {'label': 'Drive', 'value': 'Front-wheel drive'}, {'label': 'Height (mm)', 'value': 1465}, {'label': 'Engine Cylinders', 'value': 4}, {'label': 'Width (mm)', 'value': 1814}, {'label': 'Fuel System', 'value': 'Diesel Commonrail'}, {'label': 'Engine Torque', 'value': 230}, {'label': 'Engine Position', 'value': 'Front, transversely'}, {'label': 'Fuel Capacity', 'value': 50}, {'label': 'Front Breaks', 'value': 'Ventilated discs'}, {'label': 'Rear Breaks', 'value': 'Disc'}, {'label': 'Max Speed (km/h)', 'value': 183}, {'label': 'ABS', 'value': 1}, {'label': 'Engine Compression Ratio', 'value': 16.2}, {'label': 'Steering Type', 'value': 'Steering rack'}, {'label': 'Fuel Consumption l/100km (Combined)', 'value': 4.1}, {'label': 'Number of Gears', 'value': 5}, {'label': 'Engine Turbine', 'value': 'Turbo / Intercooler (Turbocharging / Intercooler)'}]
        l = map(lambda conf: (conf["label"], conf["value"]), j)
        d = dict(l)
        self.make = d['Make']
        self.model = d['Model']
        self.year= d['Model Year']
        return dict(l)


    # [{'label': 'Make', 'value': 'Skoda'}, {'label': 'Manufacturer', 'value': 'Skoda Auto AS'}, {'label': 'Plant Country', 'value': 'Czech Republic'}, {'label': 'Product Type', 'value': 'Passenger car'}, {'label': 'Manufacturer Address', 'value': 'ECO b.c.1, 293 60 Mlada Boleslav, Czech Republic'}, {'label': 'Check Digit', 'value': 'X'}, {'label': 'Sequential Number', 'value': '275210'}, {'label': 'Model', 'value': 'Octavia'}, {'label': 'Model Year', 'value': 2016}, {'label': 'Body', 'value': 'Wagon'}, {'label': 'Number of Seats', 'value': 5}, {'label': 'Number of Doors', 'value': 5}, {'label': 'Weight Empty (kg)', 'value': 1327}, {'label': 'Wheelbase (mm)', 'value': 2686}, {'label': 'Wheel Size', 'value': '195/65 R15 91H '}, {'label': 'Fuel Type - Primary', 'value': 'Diesel'}, {'label': 'Engine Displacement (ccm)', 'value': 1598}, {'label': 'Engine Power (kW)', 'value': 67}, {'label': 'Transmission', 'value': 'Manual/Standard'}, {'label': 'Series', 'value': 'ACCXXAX0 NFM5FM5A4*'}, {'label': 'Color', 'value': 'White'}, {'label': 'Registered', 'value': '2016-06-09'}, {'label': 'Engine Cylinders Position', 'value': 'Inline'}, {'label': 'Drive', 'value': 'Front-wheel drive'}, {'label': 'Height (mm)', 'value': 1465}, {'label': 'Engine Cylinders', 'value': 4}, {'label': 'Width (mm)', 'value': 1814}, {'label': 'Fuel System', 'value': 'Diesel Commonrail'}, {'label': 'Engine Torque', 'value': 230}, {'label': 'Engine Position', 'value': 'Front, transversely'}, {'label': 'Fuel Capacity', 'value': 50}, {'label': 'Front Breaks', 'value': 'Ventilated discs'}, {'label': 'Rear Breaks', 'value': 'Disc'}, {'label': 'Max Speed (km/h)', 'value': 183}, {'label': 'ABS', 'value': 1}, {'label': 'Engine Compression Ratio', 'value': 16.2}, {'label': 'Steering Type', 'value': 'Steering rack'}, {'label': 'Fuel Consumption l/100km (Combined)', 'value': 4.1}, {'label': 'Number of Gears', 'value': 5}, {'label': 'Engine Turbine', 'value': 'Turbo / Intercooler (Turbocharging / Intercooler)'}]


def main():
    car_info =  Car_Info()
    # car_info.get_car_info_fake("TMBJF7NEXG0275210")
    # print(car_info.make)
    case_dict = {}
    with open('final_legend_VINs.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
                continue
            # print(row[0], row[1].split("/"))
            if row[0] not in case_dict:
                case_dict[row[0]] = {}
                vins = row[1].split("/")
                for vin in vins:
                    try:
                        case_dict[row[0]][vin] = car_info.get_car_info_from_vin(str(vin))    # TODO : running real one
                        # print(case_dict[row[0]][vin])
                    except:
                        print(row[0], vin)
            line_count += 1
        print(f'Processed {line_count} lines.')

    print(len(case_dict.items()))

    with open("case_vin.json", 'w') as fp:
        json.dump(case_dict, fp, indent=4)

    with open("case_vin.pickle","wb") as p:
        pickle.dump(case_dict, p)

    # "decode": [
    #     "Make",
    #     "Manufacturer",
    #     "Plant Country",
    #     "Manufacturer Address",
    #     "Model Year",
    #     "Sequential Number",
    #     "Model",
    #     "Body",
    #     "Drive",
    #     "Color",
    #     "Number of Seats",
    #     "Number of Doors",
    #     "Fuel Type - Primary",
    #     "Engine Displacement (ccm)",
    #     "Engine Power (kW)",
    #     "Transmission",
    #     "Made",
    #     "Registered",
    #     "Engine (full)",
    #     "Transmission (full)",
    #     "Number of Gears",
    #     "Steering",
    #     "Check Digit",
    #     "Equipment",
    #     "Price",
    #     "Odometer (km)"
    # ]