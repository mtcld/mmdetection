import numpy as np
import cv2
from numpy import linalg as LA

#list carpart left_right
capart_left_right=["fli_fog_light","sli_side_turn_light","mirror","bpn_bed_panel","mpa_mid_panel","fender","hli_head_light","door","lli_low_bumper_tail_light","tyre","alloy_wheel","rocker_panel","qpa_quarter_panel","tli_tail_light","fbe_fog_light_bezel"]

#list carpart front_back
capart_front_back=["door","tyre","alloy_wheel"]

font_view = ['fli_fog_light','sli_side_turn_light','mirror','fbu_front_bumper','grille','fender','hli_head_light','hood','hpa_header_panel','fbe_fog_light_bezel']
back_view =['bpn_bed_panel','tail_gate','mpa_mid_panel','rbu_rear_bumper','lli_low_bumper_tail_light','trunk','hsl_high_mount_stop_light','qpa_quarter_panel','rpa_rear_panel','tli_tail_light']


def get_carpart_to_view_dimention(center_capart,view_center_list):
    score=0
    if(len(view_center_list)==0):
        return score
    center_capart = np.asarray(center_capart)
    for view_center  in view_center_list:
        view_center = np.asarray(view_center)
        score = score + np.exp(-LA.norm(center_capart - view_center))
    
    return score/len(view_center_list)

def get_side_carpart(real_carpart_center_list,view):

    m_real_carpart_center_list=[info[:] for info in real_carpart_center_list]
    #check left right
    for carpart_lr in capart_left_right:
        index_list = [i for i, value in enumerate(m_real_carpart_center_list) if (value[0] == carpart_lr)]
        if (len(index_list) == 0):
            continue
        
        if (int(view) >= 0 and int(view) <= 180):
            nv_carpart_view=["tli_tail_light","qpa_quarter_panel","lli_low_bumper_tail_light"]
            prior_label1="+r"
            prior_label2="+l"
        else:
            nv_carpart_view = ["hli_head_light", "fbe_fog_light_bezel", "mirror","fli_fog_light"]
            prior_label1 = "+l"
            prior_label2 = "+r"

        if ((carpart_lr in capart_front_back) or len(index_list) == 1):
            for index in index_list:
                if(int(view)==0 and m_real_carpart_center_list[index][1] >0.5 ):
                    m_real_carpart_center_list[index][0] =m_real_carpart_center_list[index][0]+prior_label2
                    continue

                m_real_carpart_center_list[index][0] =m_real_carpart_center_list[index][0]+prior_label1
        else:
            index1, index2 = index_list[:2]
            center_x1 = m_real_carpart_center_list[index1][1]
            center_x2 = m_real_carpart_center_list[index2][1]

            if ((center_x1 < center_x2 and (carpart_lr not in nv_carpart_view)) or (center_x1 > center_x2 and (carpart_lr  in nv_carpart_view))):
                
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + prior_label1
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + prior_label2
            else:
            
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + prior_label2 
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + prior_label1
    

    font_center_list = [value[1:] for i, value in enumerate(real_carpart_center_list) if (value[0] in font_view)]

    back_center_list = [value[1:] for i, value in enumerate(real_carpart_center_list) if (value[0] in back_view)]

    #check font back
    for carpart_lr in capart_front_back:
        index_list = [i for i, value in enumerate(real_carpart_center_list) if (value[0] == carpart_lr)]
        if (len(index_list) == 0):
            continue
            
        fb_score_list=[]
        for index in index_list:

            center=m_real_carpart_center_list[index][1:]
            font_score = get_carpart_to_view_dimention(center,font_center_list)
            back_score = get_carpart_to_view_dimention(center, back_center_list)
            fb_score_list.append([index,font_score,back_score])


        if (len(index_list) == 1):
            index,font_score, back_score=fb_score_list[0]
            if(font_score > back_score):
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "f"
            else:
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "b"
        else :
            index1,font_score1, back_score1 = fb_score_list[0]
            index2,font_score2, back_score2 = fb_score_list[1]

            if (font_score1 > font_score2 or  back_score1 <  back_score2):
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + "f"
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + "b"
            else :
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + "b"
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + "f"


    return m_real_carpart_center_list

def overlay_class_names(image, scores,labels,boxes):

    template = "{}: {:.2f}"
    height, width = image.shape[:2]

    exited_text_bbox_list = []
    for box, score, label in zip(boxes, scores, labels):
        s = template.format(get_part_name(label), score)

        text_bbox = get_text_bbox(box, s)

        x, y, new_text_bbox = get_position_of_text(text_bbox, exited_text_bbox_list, width, height)

        exited_text_bbox_list.append(new_text_bbox)
        cv2.putText(
            image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )
    return image

def get_text_bbox( boxes, score_label):
    text_size = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
    x, y = boxes[:2]
    return [x, y - text_size[0][1], text_size[0][0], text_size[0][1]]  # format x,y,w,h

def check_2_bbox_overlap( bbox1, bbox2):
    bbox1_center = [bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0]
    bbox2_center = [bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0]

    if (abs(bbox1_center[0] - bbox2_center[0]) < (bbox1[2] + bbox2[2]) and abs(
            bbox1_center[1] - bbox2_center[1]) < (bbox1[3] + bbox2[3])):
        return True
    return False

def get_position_of_text(text_bbox, exited_text_bbox_list, width, height):
    x = min(text_bbox[0] + text_bbox[2], width) - text_bbox[2]
    y = max(text_bbox[1] + text_bbox[3], 0) + text_bbox[3]

    while True:
        new_text_bbox = [x, y - text_bbox[3], text_bbox[2], text_bbox[3]]
        overlap_y_list = [(old_text_bbox[1]) for old_text_bbox in exited_text_bbox_list if
                          (check_2_bbox_overlap(new_text_bbox, old_text_bbox) == True)]

        if (len(overlap_y_list) > 0):
            y = min(max(overlap_y_list) + 3 * text_bbox[3], height - 4)
        if (len(overlap_y_list) == 0 or y == height - 4):
            break

    return x, y, new_text_bbox

def get_part_name(car_part):
    token = str(car_part).split("_")
    if len(token) == 1 or len(token[0]) > 3:
        return " ".join(token)
    else:
        return " ".join(token[1:])






