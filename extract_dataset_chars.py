import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from glob import glob


def parse_segm_points(segm_element):
    """Parses <point-x>x,y</point-x> into a numpy array of coordinates."""
    points = []
    for point_elem in segm_element:
        if point_elem.tag.startswith('point-'):
            x_str, y_str = point_elem.text.split(',')
            points.append([int(float(x_str)), int(float(y_str))])
    return np.array(points, dtype=np.int32)


def calculate_radiometrics(image, mask):
    """Calculates mean intensity and variance for a given masked region."""
    if np.count_nonzero(mask) == 0:
        return np.nan, np.nan
    pixels = image[mask == 255]
    return np.mean(pixels), np.var(pixels)


def calculate_scr(target_mean, bg_mean):
    """Calculates Signal-to-Clutter Ratio in dB."""
    if pd.isna(target_mean) or pd.isna(bg_mean) or bg_mean < 1e-6:
        return np.nan
    # Standard amplitude SCR: 20 * log10(Target_Mean / BG_Mean)
    return 20 * np.log10(target_mean / bg_mean)


def extract_ssdd_characteristics(base_dirs):
    """
    base_dirs: dictionary containing splits, e.g.,
    {'train': ('train/images', 'train/Annotations'), 'test': ...}
    """
    records = []

    for split, categories in base_dirs.items():
        # Iterate through categories (inshore, offshore, unspecified)
        for location_type, (img_dir, xml_dir) in categories.items():
            print(f"Processing {split} split -> {location_type}...")

            xml_files = glob(os.path.join(xml_dir, '*.xml'))

            for xml_path in xml_files:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                filename = root.find('filename').text
                img_path = os.path.join(img_dir, filename)

                # Image dimensions
                size_elem = root.find('size')
                img_w = int(size_elem.find('width').text)
                img_h = int(size_elem.find('height').text)

                # Load image for radiometrics
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_loaded = image is not None

                # 1. Parse all objects first to compute Nearest Neighbor Distances
                objects_data = []
                for obj in root.findall('object'):
                    obj_dict = {}
                    obj_dict['name'] = obj.find('name').text
                    obj_dict['truncated'] = int(obj.find('truncated').text)
                    obj_dict['difficult'] = int(obj.find('difficult').text)

                    # Standard Bounding Box
                    bndbox = obj.find('bndbox')
                    obj_dict['xmin'] = float(bndbox.find('xmin').text)
                    obj_dict['ymin'] = float(bndbox.find('ymin').text)
                    obj_dict['xmax'] = float(bndbox.find('xmax').text)
                    obj_dict['ymax'] = float(bndbox.find('ymax').text)
                    obj_dict['bbox_w'] = float(bndbox.find('bbox_w').text)
                    obj_dict['bbox_h'] = float(bndbox.find('bbox_h').text)

                    # Rotated Bounding Box
                    r_bndbox = obj.find('rotated_bndbox')
                    if r_bndbox is not None:
                        obj_dict['r_cx'] = float(r_bndbox.find('rotated_bbox_cx').text)
                        obj_dict['r_cy'] = float(r_bndbox.find('rotated_bbox_cy').text)
                        obj_dict['r_w'] = float(r_bndbox.find('rotated_bbox_w').text)
                        obj_dict['r_h'] = float(r_bndbox.find('rotated_bbox_h').text)
                        obj_dict['r_theta'] = float(r_bndbox.find('rotated_bbox_theta').text)

                    # Segmentation points
                    segm = obj.find('segm')
                    if segm is not None:
                        obj_dict['seg_points'] = parse_segm_points(segm)

                    objects_data.append(obj_dict)

                    # --- 2. Compute Distances and Radiometrics ---
                    for i, obj in enumerate(objects_data):
                        record = {
                            'image_path': filename,
                            'dataset': 'SSDD',
                            'split': split,
                            'location_type': location_type,  # <--- NEW COLUMN ADDED HERE
                            'truncated': obj['truncated'],
                            'difficult': obj['difficult'],
                            'image_w': img_w,
                            'image_h': img_h,
                            'image_total_pixels': img_w * img_h
                        }

                        # Normalize BBox (YOLO format: cx, cy, w, h in [0, 1])
                        n_xmin = obj['xmin'] / img_w
                        n_ymin = obj['ymin'] / img_h
                        n_xmax = obj['xmax'] / img_w
                        n_ymax = obj['ymax'] / img_h

                        record['gt_bbox'] = np.array([n_xmin, n_ymin, n_xmax, n_ymax], dtype=np.float32)
                        record['label_bbox'] = f"[{obj['xmin']:.1f}, {obj['ymin']:.1f}, {obj['xmax']:.1f}, {obj['ymax']:.1f}]"

                        record['bbox_w'] = obj['bbox_w']
                        record['bbox_h'] = obj['bbox_h']
                        record['bbox_area'] = obj['bbox_w'] * obj['bbox_h']
                        record['bbox_aspect'] = max(obj['bbox_w'], obj['bbox_h']) / (min(obj['bbox_w'], obj['bbox_h']) + 1e-6)

                        # Nearest Neighbor calculation
                        distances = []
                        for j, other_obj in enumerate(objects_data):
                            if i != j and 'r_cx' in obj and 'r_cx' in other_obj:
                                dist = np.sqrt((obj['r_cx'] - other_obj['r_cx']) ** 2 + (obj['r_cy'] - other_obj['r_cy']) ** 2)
                                distances.append(dist)
                        record['nearest_neighbor_px'] = min(distances) if distances else -1

                        # Distance to edge
                        record['dist_to_edge_px'] = min(obj['xmin'], obj['ymin'], img_w - obj['xmax'], img_h - obj['ymax'])

                        # Rotated BBox data
                        if 'r_w' in obj:
                            record['rotated_bbox_theta'] = obj['r_theta']
                            record['rotated_bbox_w'] = obj['r_w']
                            record['rotated_bbox_h'] = obj['r_h']
                            record['rotated_bbox_area'] = obj['r_w'] * obj['r_h']
                            record['rotated_bbox_aspect'] = max(obj['r_w'], obj['r_h']) / (min(obj['r_w'], obj['r_h']) + 1e-6)

                        if img_loaded:
                            # --- RADIOMETRICS: STANDARD BBOX ---
                            # Inner Mask
                            bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            x1, y1 = int(obj['xmin']), int(obj['ymin'])
                            x2, y2 = int(obj['xmax']), int(obj['ymax'])
                            cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

                            # Proportional Background Ring (Expand by 50% of w/h)
                            pad_w, pad_h = int(obj['bbox_w'] * 0.5), int(obj['bbox_h'] * 0.5)
                            bx1, by1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                            bx2, by2 = min(img_w, x2 + pad_w), min(img_h, y2 + pad_h)

                            bbox_bg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                            cv2.rectangle(bbox_bg_mask, (bx1, by1), (bx2, by2), 255, -1)
                            bbox_bg_mask = cv2.bitwise_xor(bbox_bg_mask, bbox_mask)  # Subtract inner box

                            b_mean, b_var = calculate_radiometrics(image, bbox_mask)
                            bbg_mean, bbg_var = calculate_radiometrics(image, bbox_bg_mask)

                            record['bbox_intensity'] = b_mean
                            record['bbox_variance'] = b_var
                            record['bbox_bg_intensity'] = bbg_mean
                            record['bbox_bg_variance'] = bbg_var
                            record['bbox_SCR'] = calculate_scr(b_mean, bbg_mean)

                            # --- RADIOMETRICS: SEGMENTATION MASK ---
                            if 'seg_points' in obj:
                                seg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                                cv2.fillPoly(seg_mask, [obj['seg_points']], 255)

                                # Size-dependent Morphological Dilation for SAR Background
                                # A larger ship needs a larger kernel to push past its side-lobes
                                ship_area = cv2.countNonZero(seg_mask)
                                kernel_size = max(3, int(np.sqrt(ship_area) * 0.2))  # Scale dynamically
                                if kernel_size % 2 == 0: kernel_size += 1  # Must be odd
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                                # Dilate twice to create an annular ring
                                dilated_mask = cv2.dilate(seg_mask, kernel, iterations=2)
                                seg_bg_mask = cv2.bitwise_xor(dilated_mask, seg_mask)

                                s_mean, s_var = calculate_radiometrics(image, seg_mask)
                                sbg_mean, sbg_var = calculate_radiometrics(image, seg_bg_mask)

                                record['seg_area'] = ship_area
                                record['seg_intensity'] = s_mean
                                record['seg_variance'] = s_var
                                record['seg_bg_intensity'] = sbg_mean
                                record['seg_bg_variance'] = sbg_var
                                record['seg_SCR'] = calculate_scr(s_mean, sbg_mean)

                        records.append(record)
    df_chars = pd.DataFrame(records)
    df_chars['temp_hash_key'] = df_chars['gt_bbox'].apply(lambda x: str(x.tolist()))
    df_chars = df_chars.drop_duplicates(subset=['image_path', 'temp_hash_key'], keep='last').reset_index(drop=True)
    df_chars = df_chars.drop(columns=['temp_hash_key'])
    df_chars.to_feather('sarship_dataset_chars_SSDD_OBB_Loc_Cleaned.feather')
    print(f"Extraction complete. Saved {len(df_chars)} ship records.")



if __name__ == "__main__":
    # Adjust these paths to your actual SSDD directory structure
    dataset_dirs = {
        'train': {
            'unspecified': ('/home/breandan/sarship/SSDD/train/images', '/home/breandan/sarship-irt/SSDD/Annotations_train')
        },
        'validation': {
            'unspecified': ('/home/breandan/sarship/SSDD/validation/images', '/home/breandan/sarship-irt/SSDD/Annotations_val')
        },
        'test': {
            'inshore': ('/home/breandan/sarship/SSDD/test/images', '/home/breandan/sarship-irt/SSDD/Annotations_test_inshore'),
            'offshore': ('/home/breandan/sarship/SSDD/test/images', '/home/breandan/sarship-irt/SSDD/Annotations_test_offshore')
        }
    }

    # Generate and save the characteristics CSV
    extract_ssdd_characteristics(dataset_dirs)

