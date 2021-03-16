import pickle
import os
import json
from app import get_quote, get_treatment
from external_api.vin import Car_Info

def final_list(l1):
    dc = {}
    out_ls = []
    for i in range(len(l1)):
        if l1[i][0] not in dc.keys():
            dc[l1[i][0]] = 0
        dc[l1[i][0]] = max(l1[i][1], dc[l1[i][0]])
    for i in dc.keys():
        out_ls.append({"type": i, "confidence": dc[i]})
    return out_ls


def get_all_child_folder_name(in_val):
    child_folders = set([list(val.keys())[0] for val in in_val])
    return child_folders

def damage_info_from_forder(child_folder, in_val, caseId_vin, vin="N/A"):
    pair = {}
    out_dict = {}
    for i in range(len(in_val)):
        for folder_number, data in in_val[i].items():
            if folder_number != child_folder:
                continue
            dam_val = data['damage']
            part_keys = list(data['damage'].keys())
            for j in part_keys:
                if j in ["alloy_wheel", "tyre"]:
                    continue
                if j not in pair.keys():
                    pair[j] = []
                dam_num = len(dam_val[j])
                for k in range(dam_num):
                    # 0.7, 0.8, 0.9 , 0
                    # already do in process_.py/
                    if (dam_val[j][k][0] == "scratch" and dam_val[j][k][1] >= 0) or \
                            (dam_val[j][k][0] == "loose" and dam_val[j][k][1] >= 0) or \
                            (dam_val[j][k][0] == "dent" and dam_val[j][k][1] >= 0) or \
                            (dam_val[j][k][0] == "crack" and dam_val[j][k][1] >= 0):
                        pair[j].append([dam_val[j][k][0], dam_val[j][k][1]])
                if not pair[j]:
                    del pair[j]

    out_list = []
    for i in pair.keys():
        damages = list(map(lambda damage: damage[0], pair[i]))
        treatment = get_treatment(damages)
        out_list.append({'part': i, 'damage': final_list(pair[i]), 'treatment': treatment})
    #print(out_list)
    out_dict[child_folder] = []
    # add vin
    vins = caseId_vin.get(child_folder)
    if vins is None:
        if vin == "N/A":
            vins = {"N/A": "N/A"}
        else:
            car_info = Car_Info().get_car_info_from_vin(str(vin).strip())
            if car_info:
                vins = {vin: car_info}
            else:
                vins = {vin: "N/A"}

    if vins:
        for vin in vins.keys():
            quotes = 0
            for part in out_list:
                quotes += get_quote(part['part'], part['treatment'])
            if vin == "N/A" or vins[vin] == "N/A":
                quotes_str = "N/A"
            else:
                quotes_str = str(quotes)
            out_dict[child_folder].append(
                {
                 # "vin": vin,
                 # "car-info": vins.get(vin),
                 "parts": out_list
                 # "pricing(€)": quotes_str
                }
            )

    # add car parts + damage
    # out_dict[child_folder]['parts'] = out_list
    #print(out_dict)
    return out_dict


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def main():
    pickle_data = {"carsome": "carsome"}
    folder_name = '/home/dev/dataset/carsome/test'

    with open(os.path.join("../", "external_api/case_vin.json")) as f:
        caseId_vin = json.load(f)

    for p, d in pickle_data.items():
        pickle_in = open(os.path.join("../", "dataset/KBC/pickle_file/%s.pickle" % p), "rb")
        example_dict = pickle.load(pickle_in)
        in_dict = eval(example_dict)

        in_val = in_dict[folder_name]
        child_folders = get_all_child_folder_name(in_val)

        out_put_list = {}
        for child_folder in child_folders:
            out_dict = damage_info_from_forder(child_folder, in_val, caseId_vin)
            # out_put_list.append(out_dict)
            out_put_list = {**out_put_list, **out_dict}
        # print(out_put_list)

        result_folder = os.path.join(os.getcwd(), "results")
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        with open(os.path.join(result_folder, p + "_result.json"), 'w') as fs:
            json.dump(out_put_list, fs, indent=4, default=set_default)

