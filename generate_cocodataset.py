import skimage.io as io
from pycocotools.coco import COCO
from pycocotools.cocostuffhelper import *
import os

categories = ['person','tie']
data_dir_source_coco = '/gpfs/home/cj3272/56/APPRANTI/cj3272/dataset/coco/'
data_type_source_coco = 'val2017'
data_type_source_coco = 'train2017'
#data_type_source_coco = 'test2017'
ann_file_source_coco = '{}/annotations/instances_{}.json'.format(data_dir_source_coco, data_type_source_coco)
# initialize COCO api for instance annotations
coco = COCO(ann_file_source_coco)


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


output_dir = '/gpfs/home/cj3272/56/APPRANTI/cj3272/dataset/coco_gen_12MARS2019/'
images_output_dir = output_dir + data_type_source_coco + '/images/'
masks_output_dir = output_dir + data_type_source_coco + '/masks/'
if not os.path.exists(images_output_dir):
    os.makedirs(images_output_dir)
if not os.path.exists(masks_output_dir):
    os.makedirs(masks_output_dir)

cat_ids = coco.getCatIds(catNms=categories)
img_ids = coco.getImgIds(catIds=cat_ids)

print(str(len(img_ids)) + ' images... Well done.')

for i in range(0, len(img_ids)):
    img_id = img_ids[i]
    img = coco.loadImgs(ids=img_id)
    assert 1 == len(img)
    assert img_id == img[0]['id']

    annIds = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(annIds)

    sanity_check = []
    for ann in anns:
        ann_cat_id = ann['category_id']
        assert ann_cat_id not in sanity_check
        sanity_check.append(ann_cat_id)

    mask_filename = img[0]['file_name'].replace('.jpg', '')
    if 0 == (i % 600):
        print('Exporting mask and image, processing %d of %d: %s to [%s,%s]' % (i + 1, len(img_ids), mask_filename, masks_output_dir, images_output_dir))
    segmentationPath = '%s/%s.png' % (masks_output_dir, mask_filename)
    cat_id = anns[0]['category_id']
    label_map = cocoSegmentationToSegmentationMap(coco, img_id, cat_id, check_unique_pixel_label=check_unique_pixel_label, include_crowd=include_crowd)
    #cocoSegmentationToPng(coco, img_id, cat_id, segmentationPath, check_unique_pixel_label=False)

    the_image = io.imread('%s/%s/%s' % (data_dir_source_coco, data_type_source_coco, img[0]['file_name']))
    #io.imsave('%s/%s' % (images_output_dir, img[0]['file_name']), the_image)
