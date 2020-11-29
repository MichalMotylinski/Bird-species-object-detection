# coding: utf-8

import json
import xmltodict
import os
import argparse
import pkg_resources

from bottle import (HTTPResponse, route, run, static_file, request,
                    response, Bottle, hook, get, parse_date)
import base64
from xml.sax import saxutils
import glob2
import bs4
import os
import re
import sys
import pathlib
import mimetypes
import time
from io import BytesIO as IO
import posixpath
import PIL
from PIL import Image, ImageDraw
import pickle


from renom_tag import ERROR, IMG_STATUS, NOTICE, Task
app = Bottle()
DIR_ROOT = 'public'
IMG_DIR = 'dataset'
LABEL_DIR = pathlib.Path('label')
LABEL_DETECTION = LABEL_DIR / 'detection'
LABEL_SEGMENTATION = LABEL_DIR / 'segmentation'
LABEL_SEGMENTATION_PNG = LABEL_SEGMENTATION / 'png'
LABEL_SEGMENTATION_XML = LABEL_SEGMENTATION / 'xml'

MAX_FOLDER_NAME = 256
SAVE_JSON_FILE_PATH = "label_candidates.json"


def get_VOC_palette():
    VOC_palette_path = pkg_resources.resource_filename(__name__, "VOC_palette.p")
    VOC_palette = pickle.load(open(VOC_palette_path, "rb"))
    return VOC_palette


VOC_palette = get_VOC_palette()


def create_directories(userfolder):
    dirs = [IMG_DIR, LABEL_DETECTION, LABEL_SEGMENTATION_PNG, LABEL_SEGMENTATION_XML]
    if not os.path.exists(userfolder):
        os.makedirs(userfolder)

    for d in dirs:
        joinedfolder = pathlib.Path(userfolder) / d
        joinedfolder.mkdir(exist_ok=True, parents=True)


def ensure_folder(username):
    userfolder = os.path.join(DIR_ROOT, username)
    if not os.path.exists(userfolder):
        raise ValueError('Invalid folder')
    else:
        create_directories(userfolder)


def strip_path(filename):
    if os.path.isabs(filename):
        raise ValueError('Invalid path')
    if '..' in filename:
        raise ValueError('Invalid path')
    if ':' in filename:
        raise ValueError('Invalid path')

    filename = filename.strip().strip('./\\')
    return filename


def strip_foldername(userfolder):
    if (os.path.isabs(userfolder) or
        re.search(r'[^a-zA-Z0-9_.-]', userfolder) or
            (len(userfolder) >= MAX_FOLDER_NAME)):

        raise ValueError('Invalid path')

    return userfolder


def set_json_body(body):
    response.status = 200
    response.set_header('Content-Type', 'application/json')
    return body


def filter_datafilenames(dir, ext):
    # joining public/user/dataset/
    dir = os.path.join(DIR_ROOT, os.path.normpath(dir))

    # add / in the end of "dir" if not exists
    if not dir.endswith(os.path.sep):
        dir = dir + os.path.sep

    # only a-z A-Z 0-9 _ can use as filename
    # initialze matchObject
    isvalid = re.compile(r"^[a-zA-Z0-9_.\%s]+$" % os.path.sep).match

    # path for any named files of file-extention=ext
    path = os.path.join(dir, "**", '*.' + ext)

    # 1) for name in glob2.glob(path)
    # 2) if isvalid(name)
    # 3) name[len(dir):]
    # 4) add "name" to list "names"
    #names = [name[len(dir):] for name in glob2.glob(path) if isvalid(name)]
    names = []
    undef_names = []
    for name in glob2.glob(path):
        # all files which exist in the "path"
        if isvalid(name):
            # name[len(dir):] does extract only filename
            names.append(name[len(dir):])

        if not isvalid(name):
            undef_names.append(name[len(dir):])

    return names, undef_names

# extract not-load files from def_files


def filter_duplicate_filenames(filename_list, exts):
    # 1. Extract duplicate names
    filenames_no_ext = []
    for name in filename_list:
        only_name = os.path.splitext(name)[0]
        filenames_no_ext.append(only_name)

    duplication_list = []
    for name in filenames_no_ext:
        if filenames_no_ext.count(name) >= 2:
            duplication_list.append(name)
    duplication_set = list(set(duplication_list))

    # 2. Allocate same_name_files "load" or "not_load"
    not_load_files = []
    for i in range(len(duplication_set)):
        # 1) Get files which have same name from duplication_set
        same_name_files = []
        for j in range(len(filename_list)):
            filename = os.path.splitext(str(filename_list[j]))[0]
            compare = str(duplication_set[i])
            if compare == filename:
                same_name_files.append(filename_list[j])

        # 2) Choose file which would be loaded
        load_this = ''
        for l in same_name_files:
            if exts[0] in l:
                load_this = l
                break
            elif exts[1] in l:
                load_this = l
                break
            elif exts[2] in l:
                load_this = l
                break
            elif exts[3] in l:
                load_this = l
                break

        # 3) Choose file which would "not" be loaded
        for l in same_name_files:
            if not load_this == l:
                not_load_files.append(l)

    # 4) Take the difference between "not_load_files" and the original filename_list
    load_files = list(set(filename_list) - set(not_load_files))
    load_files = sorted(load_files)
    not_load_files = sorted(not_load_files)

    return load_files, not_load_files


def get_userfolder_files(userfolder):
    ensure_folder(userfolder)
    exts = ["jpg", "jpeg", "png", "bmp"]
    ret_def_files = []
    ret_undef_files = []

    # joining user/dataset/
    dir = os.path.join(userfolder, IMG_DIR)
    for e in exts:
        def_files, undef_files = filter_datafilenames(dir, e)
        ret_def_files.extend(def_files)
        ret_undef_files.extend(undef_files)

    ret_load_files, ret_not_load_files = filter_duplicate_filenames(ret_def_files, exts)

    return ret_load_files, ret_not_load_files, ret_undef_files


def get_xml_files(userfolder):
    ensure_folder(userfolder)
    dir = os.path.join(userfolder, LABEL_DETECTION)
    return filter_datafilenames(dir, "xml")


def _get_file_name(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def get_difference_set(userfolder):
    userfolder = strip_foldername(userfolder)

    img_paths, undef_img_paths = get_img_files(userfolder)
    xml_paths, undef_xml_paths = get_xml_files(userfolder)
    xml_names = list(map(_get_file_name, xml_paths))

    def difference_set_paths_filter(img_path):
        imgname = _get_file_name(img_path)
        return imgname not in xml_names

    ret = list(filter(difference_set_paths_filter, img_paths))
    ret.sort()
    return ret


def json2xml(json_obj, line_padding=""):
    result_list = list()

    json_obj_type = type(json_obj)

    if json_obj_type is list:
        for sub_elem in json_obj:
            result_list.append(json2xml(sub_elem, line_padding))

        return "\n".join(result_list)

    if json_obj_type is dict:
        for tag_name in json_obj:
            if not re.match('^[0-9a-zA-Z]+$', tag_name):
                raise ValueError('Invalid tag name')

            sub_obj = json_obj[tag_name]
            result_list.append("%s<%s>" % (line_padding, tag_name))
            result_list.append(json2xml(sub_obj, "\t" + line_padding))
            result_list.append("%s</%s>" % (line_padding, tag_name))

        return "\n".join(result_list)

    if not isinstance(json_obj, (int, float, str)):
        raise ValueError('Invalid tag value. {}'.format(type(json_obj)))

    if isinstance(json_obj, str):
        json_obj = saxutils.escape(json_obj)

    return "%s%s" % (line_padding, json_obj)


def xml2json(xml_file, xml_attribs=True):
    with open(xml_file, "rb") as f:  # notice the "rb" mode
        d = xmltodict.parse(f, xml_attribs=xml_attribs)
        return json.dumps(d, indent=4)


def _get_resource(path, filename):
    filename = strip_path(filename)
    body = pkg_resources.resource_string(__name__, posixpath.join('.build', path, filename))

    headers = {}
    mimetype, encoding = mimetypes.guess_type(filename)
    if mimetype:
        headers['Content-Type'] = mimetype
    if encoding:
        headers['encoding'] = encoding
    return HTTPResponse(body, **headers)


@app.route("/")
def index():
    return _get_resource('', 'index.html')


@app.route("/admin")
def index():
    return _get_resource('', 'index.html')


@app.route("/static/<file_name:re:.+>")
def static(file_name):
    return _get_resource('static', file_name)


def check_path(path, filename):
    head = os.path.abspath(path)
    if not head.endswith(('/', '\\')):
        head += os.path.sep

    filename = os.path.abspath(os.path.join(head, filename))

    if not filename.startswith(head):
        raise ValueError('invalid path')

    return filename


def get_userfolder_path(userfolder):
    return check_path(DIR_ROOT, userfolder)


def get_annotation_xml(userfolder, img_filename, task_id):
    xmlfile = None
    filename = _get_file_name(img_filename) + '.xml'

    if (task_id == Task.DETECTION.value):
        xmlfolder = str(get_userfolder_path(userfolder) / LABEL_DETECTION)
    elif (task_id == Task.SEGMENTATION.value):
        xmlfolder = str(get_userfolder_path(userfolder) / LABEL_SEGMENTATION_XML)

    xmlfile = check_path(xmlfolder, filename)

    if not os.path.exists(xmlfile):
        json_dict = None
    else:
        json_data = xml2json(xmlfile)
        json_dict = json.loads(json_data)

        # revert `object` to original name(`objects`)
        json_dict['annotation']['objects'] = json_dict['annotation']['object']
        del json_dict['annotation']['object']
        try:
            #  When the item in `objects` is only one, `objects` become dict. So revert to list.
            if isinstance(json_dict['annotation']['objects'], dict):
                temp = [{"object": json_dict['annotation']['objects']}]
                json_dict['annotation']['objects'] = temp

            #  When the items in `objects` are multiple, `objects` become list but no "object" key. so add it.
            elif isinstance(json_dict['annotation']['objects'], list):
                temp = [{"object": obj} for obj in json_dict['annotation']['objects']]
                json_dict['annotation']['objects'] = temp

        except KeyError:
            json_dict['annotation']['objects'] = []

        if task_id == Task.SEGMENTATION.value:
            # unwrap <point> tag
            json_dict = wrap_unwrap_points_by_tag(json_dict)

        # None を空文字列に変換
        if not json_dict['annotation']['source'].get('reviewresult', False):
            json_dict['annotation']['source']['reviewresult'] = ''

        # Check comment
        source = json_dict['annotation']['source']
        old_comment = source.get('reviewcomment', None)
        new_comment = source.get('comment', None)
        if old_comment is not None:
            del json_dict['annotation']['source']['reviewcomment']
        if new_comment is None:
            json_dict['annotation']['source']['comment'] = {'admin': '', 'subord': ''}

    return json_dict


@app.route("/api/get_raw_img/<userfolder>/<file_name:re:.+>")
def get_raw_img(userfolder, file_name):
    filename = check_path(os.path.join(get_userfolder_path(userfolder), IMG_DIR), file_name)

    img = open(filename, "rb").read()
    encoded_img = base64.b64encode(img)
    encoded_img = encoded_img.decode('utf8')

    im = PIL.Image.open(filename)
    width, height = im.size
    ret = json.dumps({
        'img': encoded_img,
        'width': width,
        'height': height
    })

    ret = set_json_body(ret)
    return ret


@app.route("/t/<userfolder:re:.+>/<file_name:re:.+>")
def get_thumbnail(userfolder, file_name):

    filename = check_path(os.path.join(get_userfolder_path(userfolder), IMG_DIR), file_name)

    headers = {}
    stats = os.stat(filename)

    lm = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stats.st_mtime))
    response.set_header('Last-Modified', lm)

    ims = request.environ.get('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims = parse_date(ims.split(";")[0].strip())
    if ims is not None and ims >= int(stats.st_mtime):
        headers['Date'] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        return HTTPResponse(status=304, **headers)

    response.content_type = 'image/png'

    img = Image.open(filename, 'r')
    img.thumbnail((70, 70), Image.ANTIALIAS)
    buffered = IO()
    img.save(buffered, format='PNG')

    ret = buffered.getvalue()
    response.set_header('Content-Length', len(ret))

    return ret

# roothing for get_filename_obj


@app.route("/api/get_filename_obj", method="POST")
def get_filename_obj():
    userfolder = request.json['username']
    userfolder = strip_foldername(userfolder)
    task_id = request.json['task_id']

    imgname_list, dup_imgname_list, undef_imgname_list = get_userfolder_files(userfolder)

    ret = {}
    for imgname in imgname_list:
        # get xml for detction or segmentation
        annotation = get_annotation_xml(userfolder, imgname, task_id)

        # ret -> { imgname: xmlinfo, ... }
        ret[imgname] = annotation

    body = json.dumps({
        "filename_obj": ret,
        "undef_img_list": undef_imgname_list,
        "dup_img_list": dup_imgname_list
    })
    ret = set_json_body(body)
    return ret


def save_seg_png(image_size, polygons, username, filename):
    """save annotation png from list by using PIL

    :return: none
    """

    # 1. saving path
    save_png_path = get_userfolder_path(username) / LABEL_SEGMENTATION_PNG
    save_png_filename = str((save_png_path / _get_file_name(filename)).with_suffix(".png"))

    # 2. create initial_img : base canvas for drawing polygons. (first RGB)
    #    create palette_img : temporary "P" mode image which will conver initial_img as VOC_palette "P" mode image
    initial_img = Image.new('RGB', (int(image_size['width']), int(image_size['height'])), 0)
    palette_img = Image.new('P', (16, 16))

    # 3. load VOC_palette.p from source dir and add to "palimage":  temp "P" mode image
    palette_img.putpalette(VOC_palette)

    # 4. convert "RGB" initial_img_img to "P" mode by using VOC_palette image
    img = initial_img.quantize(colors=256, method=None, kmeans=0, palette=palette_img)

    # 5. draw polygons by using coordinates
    draw = ImageDraw.Draw(img)
    for object in polygons:
        points = object['object']['points']
        label_id = int(object['object']["labelid"])

        points_tuple = []
        for point in points:
            p_tuple = (point["x"], point["y"])
            points_tuple.append(p_tuple)
        draw.polygon(points_tuple, fill=label_id)

    # 6. save image as png
    img.save(save_png_filename, quality=95)


def wrap_unwrap_points_by_tag(label_dict):
    """wrap points by xml tag

    :return: converted label_dict
    """
    objects = label_dict['annotation']['objects']
    del label_dict['annotation']['objects']

    # A. wrap points by <point> tag
    if isinstance(objects[0]['object']['points'], list):
        for obj in objects:
            points = obj['object']['points']
            del obj['object']['points']
            wrapped_points = [{"point": p} for p in points]
            obj['object']['points'] = wrapped_points
            # object['object']['points'] = {}
            # for p in points :
            #     object['object']['points'].update({ "point": p })

    # B. unwrap <point> tag
    elif isinstance(objects[0]['object']['points'], dict):
        for obj in objects:
            try:
                poly = obj['object']
                if isinstance(poly['points']['point'], list):
                    points_temp = []
                    for p in poly['points']['point']:
                        p_temp = {'x': None, 'y': None}
                        p_temp['x'] = float(p['x'])
                        p_temp['y'] = float(p['y'])
                        points_temp.append(p_temp)
                    poly['points'] = points_temp

            except KeyError:
                jpoly['points'] = []

    label_dict['annotation']['objects'] = objects

    return label_dict


@app.route("/api/save_annotation", method=["POST"])
def save_annotation():
    """save xml file from dictionary

    :return:
    """

    # None check
    def none_check(d):
        for k, v in d.items():
            if v is None:
                d[k] = ''
            elif isinstance(v, dict):
                none_check(v)

    task_id = request.json['task_id']
    label_dict = request.json['value']
    username = request.json['username']
    none_check(label_dict)

    ann_path = strip_path(label_dict['annotation']['path'])
    check_path(IMG_DIR, ann_path)

    userfolder, file_name = posixpath.split(ann_path)
    userfolder = os.path.join('dataset', userfolder).rstrip('/')  # add 'dataset' for compatibility

    label_dict['annotation']['folder'] = userfolder
    label_dict['annotation']['filename'] = file_name

    if task_id == Task.DETECTION.value:
        userfolder_path = str(get_userfolder_path(username) / LABEL_DETECTION)
    elif task_id == Task.SEGMENTATION.value:
        userfolder_path = str(get_userfolder_path(username) / LABEL_SEGMENTATION_XML)
        image_size = label_dict['annotation']['size']
        polygons = label_dict['annotation']['objects']
        save_seg_png(image_size, polygons, username, file_name)

        # wrap points by <point> tag
        label_dict = wrap_unwrap_points_by_tag(label_dict)

    # get save_file_name
    save_xml_file_name = check_path(userfolder_path, _get_file_name(file_name)) + '.xml'
    # convert dict to xml
    xml_data = json2xml(label_dict)
    # extract objects
    xml_soup = bs4.BeautifulSoup(xml_data, 'lxml')

    if (xml_soup.find('object')):
        xml_soup.find('object').parent.unwrap()

    # save
    with open(save_xml_file_name, 'w') as ftpr:
        ftpr.write(xml_soup.find('annotation').prettify())

    # print('%s is saved' % (save_xml_file_name))

    label_dict = get_annotation_xml(username, file_name, task_id)
    ret = set_json_body({'result': label_dict})
    return ret


@app.route("/api/delete_annotation", method=["POST"])
def delete_annotation():
    userfolder = request.json['username']
    filename = _get_file_name(request.json['target_filename'])
    task_id = request.json['task_id']

    if task_id == Task.DETECTION.value:
        filename = filename + ".xml"
        xmldir = str(get_userfolder_path(userfolder) / LABEL_DETECTION)
        delete_xml_file_name = check_path(xmldir, filename)

    elif task_id == Task.SEGMENTATION.value:
        xml_filename = filename + ".xml"
        png_filename = filename + ".png"
        xmldir = str(get_userfolder_path(userfolder) / LABEL_SEGMENTATION_XML)
        pngdir = str(get_userfolder_path(userfolder) / LABEL_SEGMENTATION_PNG)
        delete_xml_file_name = check_path(xmldir, xml_filename)
        delete_png_file_name = check_path(pngdir, png_filename)

    xml_result = True
    png_result = True

    if os.path.exists(delete_xml_file_name):
        os.remove(delete_xml_file_name)
        xml_result = NOTICE['DELETION']['XML']['SUCCESS']['code']
        message = NOTICE['DELETION']['XML']['SUCCESS']['message']
        print(message)
        print('%s is deleted!' % (delete_xml_file_name))
    else:
        xml_result = ERROR['DELETION']['XML']['code']
        message = ERROR['DELETION']['XML']['message']
        print(message)
        print('filename:%s connot be found. Please check if the xml-file exists!' %
              (delete_xml_file_name))

    if task_id == Task.DETECTION.value:
        ret = set_json_body(json.dumps({'result': xml_result}))

    elif task_id == Task.SEGMENTATION.value:
        if os.path.exists(delete_png_file_name):
            os.remove(delete_png_file_name)
            png_result = NOTICE['DELETION']['PNG']['SUCCESS']['code']
            message = NOTICE['DELETION']['PNG']['SUCCESS']['message']
            print(message)
            print('%s is deleted!' % (delete_png_file_name))
        else:
            png_result = ERROR['DELETION']['PNG']['code']
            message = ERROR['DELETION']['PNG']['message']
            print(message)
            print('filename:%s connot be found. Please check if the png-file exists!' %
                  (delete_png_file_name))

        ret = set_json_body(json.dumps({'xml_result': xml_result, 'png_result': png_result}))

    return ret


def get_annotated_images(userfolder, task_id):

    if task_id == Task.DETECTION.value:
        xmlfolder = (DIR_ROOT / userfolder / pathlib.Path(LABEL_DETECTION))

    elif task_id == Task.SEGMENTATION.value:
        xmlfolder = (DIR_ROOT / userfolder / pathlib.Path(LABEL_SEGMENTATION_XML))

    # keep the path for xml files
    before_sort_info = []
    for p in xmlfolder.iterdir():
        if p.is_file() and str(p).endswith('.xml'):
            before_sort_info.append(p.relative_to(xmlfolder))
    # use for sort
    sort_info = []

    for annotated_img in before_sort_info:
        time = os.stat(str(xmlfolder / annotated_img)).st_mtime
        sort_info.append(dict(time=time, annotated_img=annotated_img))

    # sort by edited time
    _sort = sorted(sort_info, key=lambda x: x['time'], reverse=False)

    # remove dictionary key
    annotated_info = [item.pop('annotated_img') for item in sort_info]

    ret = []
    # get annotated images from xml
    for annotated_img in reversed(annotated_info):

        # load xml file convert to json
        # extract bounding box
        xml = get_annotation_xml(str(userfolder), str(annotated_img), task_id)
        filename = check_path(os.path.join(get_userfolder_path(str(userfolder)),
                                           IMG_DIR), xml['annotation']['filename'])

        img = open(filename, "rb").read()
        encoded_img = base64.b64encode(img)
        encoded_img = encoded_img.decode('utf8')

        # add annotated info
        ret_item = dict(
            filename=xml['annotation']['filename'],
            height=xml['annotation']['size']['height'],
            width=xml['annotation']['size']['width'],
            image="data:image;base64," + encoded_img
        )

        temp_objs = []
        objects = xml['annotation']['objects']

        if task_id == Task.DETECTION.value:
            # get objects
            for obj in objects:
                left = obj['object']['bndbox']['xmin']
                right = obj['object']['bndbox']['xmax']
                top = obj['object']['bndbox']['ymin']
                bottom = obj['object']['bndbox']['ymax']
                label = obj['object']['name']
                temp_objs.append(dict(left=left, right=right, top=top, bottom=bottom, label=label))

            ret_item["boxes"] = temp_objs

        elif task_id == Task.SEGMENTATION.value:
            for obj in objects:
                label = obj['object']['name']
                points = obj['object']['points']
                temp_objs.append(dict(points=points, label=label))

            ret_item["polygons"] = temp_objs

        ret.append(ret_item)
    return ret


@app.route("/api/load_annotated_images", method=["POST"])
def load_annotated_images():
    userfolder = request.json['username']
    userfolder = pathlib.Path(strip_foldername(userfolder))
    task_id = request.json['task_id']
    # dataset_imgfolder = (DIR_ROOT / userfolder / pathlib.Path(IMG_DIR))

    annotated_img = []
    annotated_img = get_annotated_images(userfolder, task_id)
    # ret -> {result: [{ filename:_ , boxes:_ , ... } ,{}, ...]}
    ret = set_json_body({'result': annotated_img})
    return ret


def save_seg_label_txtdata(labels, username):
    # TODO muraishi : 7. relate  the chenges of class_map.txt to the exsisting polygons information

    name_id_list = []
    # "labels" have already hold "background" label
    for key, value in labels.items():
        name = value['label']
        name_id_tup = (str(name), str(key))
        name_id_str = " ".join(name_id_tup)
        name_id_list.append(name_id_str)

    save_png_path = get_userfolder_path(username) / LABEL_SEGMENTATION_PNG
    txtfile = str((save_png_path / "class_map").with_suffix(".txt"))

    with open(txtfile, 'w') as ftpr:
        ftpr.write("\n".join(name_id_list))


@app.route("/api/save_label_candidates_dict", method=['POST'])
def save_label_candidates_dict():
    # the data is initially created in mutation.vue of client (addLabelToState)

    src_labels = request.json['labels']
    task_id = request.json['task_id']
    username = request.json['username']

    max_id = 0
    task_labels = {}
    for key, v in src_labels.items():
        label = v['label'].strip()
        shortcut = v['shortcut'].strip()

        if not re.match(r"^[0-9a-zA-Z]+$", label):
            raise ValueError("Invalkey label")
        if not shortcut:
            shortcut = 'no_shortcut%s' % key

        if 'id' in v:
            label_id = v['id']
            task_labels[label_id] = {'label': label, 'shortcut': shortcut}
            if max_id < int(label_id):
                max_id = int(label_id)

        else:  # initial addtion of a label
            if int(key) < max_id:
                key = max_id
            label_id = key
            task_labels[label_id] = {'label': label, 'shortcut': shortcut}

    final_labels_dict = {}
    userfolder_path = get_userfolder_path(username)
    jsonfile = os.path.join(userfolder_path, SAVE_JSON_FILE_PATH)

    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as ftpr:
            final_labels_dict = json.load(ftpr)
    else:
        final_labels_dict = {"DETECTION": {},
                             "SEGMENTATION": {}}

        # the format of label_candidates.json
    if task_id == Task.DETECTION.value:
        final_labels_dict["DETECTION"] = task_labels

    elif task_id == Task.SEGMENTATION.value:
        final_labels_dict["SEGMENTATION"] = task_labels
        save_seg_label_txtdata(task_labels, username)

    with open(jsonfile, 'w') as fw:
        json.dump(final_labels_dict, fw, indent=4)


@app.route("/api/load_label_candidates_dict", method=["POST"])
def load_label_candidates_dict():
    username = request.json['username']
    task_id = request.json['task_id']

    userfolder_path = get_userfolder_path(username)
    jsonfile = os.path.join(userfolder_path, SAVE_JSON_FILE_PATH)

    task_labels = {}
    if os.path.exists(jsonfile):
        with open(jsonfile, 'r') as ftpr:
            json_data = json.load(ftpr)

        task_json_data = {}

        # swith the focus in label_candidates.json by task_id
        if task_id == Task.DETECTION.value:
            task_json_data = json_data["DETECTION"]

        elif task_id == Task.SEGMENTATION.value:
            task_json_data = json_data["SEGMENTATION"]

        for k, v in sorted(task_json_data.items()):
            if 'no_shortcut' in v['shortcut']:
                v['shortcut'] = ''
            task_labels[k] = {'label': v['label'], 'shortcut': v['shortcut']}

    # TODO muraihsi: 6. for SEGMENTATION, check class_map.txt before send the information to cilent
    body = json.dumps(task_labels)
    return set_json_body(body)


@app.route("/api/delete_label_candidates_dict", method=["POST"])
def delete_label_candidates_dict():
    label_dict = request.json

    userfolder_path = get_userfolder_path(label_dict['username'])
    jsonfile = os.path.join(userfolder_path, SAVE_JSON_FILE_PATH)

    if os.path.exists(jsonfile):
        os.remove(jsonfile)


# DIR_ROOT = 'public'
# search inside the "bublic" ande get userlist
@app.route("/api/userlist", method=["POST"])
def get_userlist():
    # users = the list of user-userfolder
    current_dir = os.getcwd()
    public = os.path.join(current_dir, DIR_ROOT)

    if os.path.exists(public) and os.path.isdir(public):
        users = []
        for d in sorted(os.listdir(DIR_ROOT)):
            if not re.match(r"^[a-zA-Z0-9._]+$", d):
                continue

            if not os.path.isdir(os.path.join(DIR_ROOT, d)):
                continue

            if not os.path.exists(os.path.join(DIR_ROOT, d, IMG_DIR)):
                continue

            users.append(d)
        ret = set_json_body(json.dumps({'result': 1, 'user_list': users}))

    else:
        #message = 'No userfolder named "public" in the current directory. \n Wanna create directories?'
        #message = 'No userfolder named "public" in the current directory: \n'+ str(current_dir) + '\n'
        #message = message + 'Wanna create directories?'
        ret = set_json_body(json.dumps({'result': 0, 'user_list': None}))

    return ret


@app.route("/api/make_dir", method=["POST"])
def make_dir():
    current_dir = os.getcwd()
    username = request.json['username']
    result = NOTICE['MAKE_DIR']['INITIAL']['code']

    if not os.path.exists(current_dir):
        result = ERROR['MAKE_DIR']['NG_PATH']['code']
        message = ERROR['MAKE_DIR']['NG_PATH']['message'] + current_dir
        print(message)

    elif not re.match(r"^[a-zA-Z0-9._]+$", username):
        result = ERROR['MAKE_DIR']['NG_USERNAME']['code']
        message = ERROR['MAKE_DIR']['NG_USERNAME']['message']
        print(message)

    else:
        # string -> join to pat
        # public = os.path.join(current_dir, DIR_ROOT)
        # userfolder = os.path.join(public, username)
        userfolder = os.path.join(DIR_ROOT, username)
        create_directories(userfolder)

        result = NOTICE['MAKE_DIR']['SUCCESS']['code']
        message = NOTICE['MAKE_DIR']['SUCCESS']['message']
        print(message)

    #message = message + "load again to start."
    ret = set_json_body(json.dumps({'result': result}))
    return ret


def main():
    parser = argparse.ArgumentParser(description='ReNomTAG')
    parser.add_argument('--port', type=int, help='Port Number', default=8885)
    args = parser.parse_args()
    run(app, host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
