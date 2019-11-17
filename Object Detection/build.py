def build_dataset():
    # Begin
    train_an = "/mnt/data/cv3/train/VOCdevkit/VOC2007/Annotations/"
    train_img = "/mnt/data/cv3/train/VOCdevkit/VOC2007/JPEGImages/"
    dest = "/mnt/data/cv3/train/"
    #os.chdir("/mnt/data/cv3/train")
    for file in os.listdir(train_img):
        img_name = file.rsplit(".")[0]
        img = cv2.imread(train_img + file)
        print(img_name,end = " ")
        tree = ET.parse(train_an + img_name + '.xml')
        root = tree.getroot()
        id =0
        for child in root.iter('object'):
            id+=1
            name = child.find('name').text
            if name != 'aeroplane' and name != 'chair' and name != 'bottle':
                id-=1
                continue
            bndbox = child.find('bndbox')
            box=[]
            box.append(int(bndbox.find('xmin').text))
            box.append(int(bndbox.find('ymin').text))
            box.append(int(bndbox.find('xmax').text))
            box.append(int(bndbox.find('ymax').text))
            cv2.imwrite(dest + name + "/" + img_name + "_" + str(id)+".jpg" ,img[box[1]:box[3] ,box[0]:box[2]])
            print(id, end = " ")
        print(" ")
