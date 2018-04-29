import sys, os, csv, multiprocessing, urllib2
from PIL import Image
from StringIO import StringIO

def parseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    counts = {}
    key_url_list = [line[:3] for line in csvreader]
    key_url_list = key_url_list[1:]
    for entry in key_url_list:
        k = entry[2]
        if k in counts:
            counts[k] = counts[k] + 1
        else:
            counts[k] = 1
    max_occurences = sorted(counts.items(), key = lambda x : x[1], reverse = True)
    max_occurences = max_occurences[:10]
    max_10 = [item[0] for item in max_occurences]

    return_list = []
    max_occurences = {x[0]:0 for x in max_occurences}
    for entry in key_url_list:
        if (entry[2] in max_10) and (max_occurences[entry[2]] < 3000):
            max_occurences[entry[2]] = max_occurences[entry[2]] + 1
            return_list.append(entry[:3])
    return return_list

def downloadImage(key_url):
    (key, url, label) = key_url
    out_dir = sys.argv[2]
    out_dir = os.path.join(out_dir, label)
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(filename):
        return
    try:
        response = urllib2.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(StringIO(image_data))
    except:
        print('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb = pil_image_rgb.resize((128,128), Image.ANTIALIAS)
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return

def main():
    data_file = sys.argv[1]
    key_url_list = parseData(data_file)
    pool = multiprocessing.Pool(processes=50)
    pool.map(downloadImage, key_url_list)

if __name__ == '__main__':
    main()
