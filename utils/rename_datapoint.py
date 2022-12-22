import os

PATH = "/Users/ujjwal/Documents/data/test"

sorted_classes = sorted(os.listdir(PATH))

allowed_classes = ["10", "20", "50", "100", "200", "500", "2000"]
sorted_classes = [x for x in sorted_classes if x in allowed_classes]
print(sorted_classes)
for id, dir_name in enumerate(sorted_classes):
    for img in os.listdir(os.path.join(PATH, dir_name)):
        if img.endswith(".jpg"):
            os.rename(os.path.join(PATH, dir_name, img), os.path.join(PATH, dir_name, dir_name + "_" + img))
