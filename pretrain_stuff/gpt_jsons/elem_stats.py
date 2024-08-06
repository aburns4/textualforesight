import json, glob

from collections import defaultdict

GENERIC_WORDS = ['action', 'bar', 'menu', 'title', 'and', 'ans', 'app', 'icon', 'name',
                 'arg', 'background', 'element', 'btn', 'but', 'bottom', 'button', 'content',
                 'desc', 'text', 'item', 'empty', 'fab', 'image', 'grid', 'header', 'img',
                 'imgfile', 'lbutton', 'label', 'letter', 'list', 'view', 'pic', 'placeholder',
                 'random', 'row', 'single', 'raw', 'small', 'large', 'sub', 'template', 'navbar', 
                 'banner', 'test', 'textinput', 'error', 'texto', 'todo', 'toolbar', 'tool', 'track',
                 'txt', 'unknown', 'stub', 'web', 'left', 'right', 'tlb', 'nan', 'page', 'feature',
                 'menugrid', 'picture', 'tabs', 'number', 'node', 'iconimage', 'entity', 'webview',
                 'heading', 'logo', 'tbl', 'tab', 'primary', 'footer']

def is_good_ocr(field):
    toks = field.split(' ')
    only_gen = (len(set(toks).difference(set(GENERIC_WORDS))) == 0)
    single_or_empty_char = (len(field) <= 1)
    is_url = (len(toks) == 1 and 'http' in field)
    transformed_field = field.encode('unicode-escape').decode('ascii')
    is_alpha = all(x.isalpha() or x.isspace() for x in transformed_field)
    if (not only_gen) and (not single_or_empty_char): # and (not is_url) and is_alpha:
        return True
    return False


aitw = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/aitw/elements_raw/*")
longitudinal = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/longitudinal/elements_final/*")
motif = glob.glob("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/motif/elements_final/*")

all_files = aitw + longitudinal + motif
zero_elems = 0
lt3 = 0
num_words = []
non_zero_counts = []
kept_samples = []

for fi in all_files:
    print(fi)
    dataset = fi.split('/')[7]
    with open(fi) as f:
        data = json.load(f)
    for app in data:
        for uid in data[app]:
            elems = data[app][uid]
            if dataset == "aitw":
                final_elems = []
                for e in elems:
                    if is_good_ocr(e):
                        final_elems.append(e)
            else:
                final_elems = elems
            
            # elems = [e.lower() for e in elems] choosing to not lowercase
            x = len(final_elems)
            if x == 0:
                zero_elems += 1
            else:
                non_zero_counts.append(x)
                if x < 3:
                    lt3 += 1

                for e in final_elems:
                    num_words.append(len(e.split(" ")))
                
                text = ' | '.join(final_elems)
                assert "\n" not in text
                assert "[*]" not in text
                kept_samples.append((dataset, app, uid, text))

print('Can\'t be used due to zero elems %d, remaining samples %d' % (zero_elems, len(non_zero_counts)))
non_zero_counts.sort()
med = int(len(non_zero_counts) / 2)
avg = float(sum(non_zero_counts)) / len(non_zero_counts)

print('Remaining elems stats min %d average %.2f median %d max %d' % (min(non_zero_counts), avg, non_zero_counts[med], max(non_zero_counts)))
print('Average number of words per element %.2f' % (sum(num_words)/len(num_words)))
print('Samples with < 3 elems %d' % lt3)

kept_nouid_samples = [(x[1], x[3]) for x in kept_samples]
print('\nTotal kept samples %d, unique kept samples %d' % (len(kept_samples), len(set(kept_nouid_samples))))

with open('set_samples_all_final.txt', 'w') as f:
    to_write = ['[*]'.join(x) for x in set(kept_nouid_samples)]
    f.write('\n'.join(to_write))