import json
import glob
import os
import random

very_large = ["com.android.settings_.SubSettings",
              "com.android.chrome_com.google.android.apps.chrome.Main",
              "com.android.chrome_org.chromium.chrome.browser.preferences.Preferences",
              "com.android.vending_com.android.vending.AssetBrowserActivity",
              "com.google.android.apps.maps_com.google.android.maps.MapsActivity"]

all_app_paths = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons/aitw/aitw_by_app_thresholded/*/*.json")
app_set = set([x.split('/')[-1].split(".json")[0] for x in all_app_paths])
print(len(app_set))
for save_app_name in app_set:
    if save_app_name in very_large:
        print("SKIPPING: " + save_app_name)
        continue
    all_jsons = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons/aitw/aitw_by_app_thresholded/*/" + save_app_name + ".json")
    new_save_path = "/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons/aitw/grouped_apps/" + save_app_name 

    if os.path.exists(new_save_path + ".json"):
        print("ALREADY EXISTS... CONTINUING " + save_app_name)
        continue
    print(save_app_name)

    for j in all_jsons:
        print(j)
        with open(j) as f:
            indiv_data = json.load(f)

        if save_app_name in very_large:
            rand_choice = random.randint(0,4)
            final_path = new_save_path + str(rand_choice) + ".json"
        else:
            final_path = new_save_path + ".json"

        if os.path.exists(final_path):
            with open(final_path) as f:
                curr_samples = json.load(f)

            combined = indiv_data + curr_samples
            with open(final_path, "w") as f:
                json.dump(combined, f)
        else:
            with open(final_path, "w") as f:
                json.dump(indiv_data, f)