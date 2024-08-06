import json, glob
import os

aitw = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/aitw/elements_no_icon/*")
motif = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/motif/elements_final/*")
longitudinal = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/longitudinal/elements_final/*")

all_files = aitw + motif + longitudinal
for elem_file in all_files:
    print(elem_file)
    save_file = elem_file.replace("gpt_jsons", "spotlight_jsons").replace("elements", "elem_list_captions")
    print(save_file)
    json_l = []
    with open(elem_file) as f:
        data = json.load(f)
    for app in data:
        for imid in data[app]:
            dedup_elems = []
            for e in data[app][imid]:
                if e not in dedup_elems:
                    dedup_elems.append(e)
            caption = ", ".join(dedup_elems)
            
            if "longitudinal" in save_file:
                imid = os.path.join(app, imid + ".png")
            elif "aitw" in save_file:
                imid = imid + ".jpg"
            else:
                assert "motif" in save_file
                impath = glob.glob("/projectnb2/ivc-ml/aburns4/stage2/*/" + app + "/*/screens/" + imid + ".jpg")
                assert len(impath) > 0
                imid = impath[0].split("stage2/")[-1]
            json_l.append({"image": imid, "caption": caption})
    
    new_folder = "/".join(save_file.split("/")[:-1])
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    with open(save_file, "w") as f:
        json.dump(json_l, f)