def intersection_over_union(box1, box2):
    [x_min_1, y_min_1, x_max_1, y_max_1] = box1
    [x_min_2, y_min_2, x_max_2, y_max_2] = box2
    delta_x = min(x_max_1,x_max_2)-max(x_min_1,x_min_2)
    delta_y = min(y_max_1,y_max_2)-max(y_min_1,y_min_2)
    if(delta_x==0 || delta_y==0) : return 0
    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)
    and_area = delta_x*delta_y
    or_area = area_1+area_2-and_area
    return and_area/or_area

def sliding_window_for_background(image, list_of_boxes, threshold, step_size, window_size):
    list_of_background=[]
    for y in range(0,image.shape[0],step_size):
        for x in range(0,image.shape[1],step_size):
            slide_box = [x,y,x+window_size[0],y+window_size[1]]
            flag=0
            for i in range(len(list_of_boxes)):
                if intersection_over_union(slide_box, list_of_boxes[i])>threshold:
                    flag=1
                    break
            if flag: continue
            else : list_of_background.append(slide_box)
    return list_of_background