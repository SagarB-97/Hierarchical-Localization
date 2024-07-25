from collections import defaultdict
import h5py
import numpy as np
import pycolmap
import torch
import os
import cv2
from pathlib import Path

from .extract_features import ImageDataset, resize_image
from .utils.parsers import names_to_pair
from .utils.io import read_image
from .utils.read_write_model import read_model, write_model, Camera, Image, Point3D
from .localize_sfm import QueryLocalizer
from scipy.spatial.transform import Rotation

_default_preprocessing_conf = {
    'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
    'grayscale': False,
    'resize_max': None,
    'resize_force': False,
    'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
}

def _get_processed_image(query_dir, query_img_name, preprocessing_conf):
    preprocessing_conf = {**_default_preprocessing_conf, **preprocessing_conf}

    image = read_image(query_dir / query_img_name, preprocessing_conf['grayscale'])
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]

    if preprocessing_conf['resize_max'] and (preprocessing_conf['resize_force']
                                or max(size) > preprocessing_conf['resize_max']):
        scale = preprocessing_conf['resize_max'] / max(size)
        size_new = tuple(int(round(x*scale)) for x in size)
        image = resize_image(image, size_new, preprocessing_conf['interpolation'])

    if preprocessing_conf['grayscale']:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.

    data = {
        'image': torch.from_numpy(image),
        'original_size': torch.from_numpy(np.array(size)),
    }
    return data

def get_local_features(query_processing_data_dir, 
             query_image_name,
             local_feature_conf,
             local_features_extractor_model,
             device,
             ):
    """
    Extract local features from a query image.
    Parameters:
        query_processing_data_dir: Path to the directory containing the query images.
        query_image_name: Name of the query image.
        local_features_extractor_model: Local feature extractor model.
    """
    data = _get_processed_image(query_processing_data_dir, query_image_name, local_feature_conf['preprocessing'])
    with torch.no_grad():
        local_features = local_features_extractor_model({'image': data['image'].unsqueeze(0).to(device)})
    local_features = {k: v[0] for k, v in local_features.items()}

    local_features['image_size'] = original_size = data['original_size']
    
    # Scale keypoints
    size = np.array(data['image'].shape[-2:][::-1])
    scales = (original_size / size).to(device, dtype = torch.float32)
    local_features['keypoints'] = (local_features['keypoints'] + .5) * scales[None] - .5
    if 'scales' in local_features:
        local_features['scales'] *= scales.mean()
    # add keypoint uncertainties scaled to the original resolution
    uncertainty = getattr(local_features_extractor_model, 'detection_noise', 1) * scales.mean()
    
    return local_features, uncertainty

def get_global_descriptors(query_processing_data_dir, 
                           query_image_name, 
                           global_descriptor_conf, 
                           global_descriptor_model, 
                           device):
    """
    Extract global descriptors from a query image.
    Parameters:
        query_processing_data_dir: Path to the directory containing the query images.
        query_image_name: Name of the query image.
        global_descriptor_conf: Configuration of the global descriptor model.
        global_descriptor_model: Global descriptor model.
        device: Device to run the model on.
    """
    data = _get_processed_image(query_processing_data_dir, query_image_name, global_descriptor_conf['preprocessing'])
    with torch.no_grad():
        global_descriptor = global_descriptor_model({'image': data['image'].unsqueeze(0).to(device)})
    global_descriptor = {k: v[0] for k, v in global_descriptor.items()}
    global_descriptor['image_size'] = data['original_size'][0]
    
    return global_descriptor

def get_candidate_matches(global_descriptor, 
                          db_global_descriptors, 
                          db_image_names):
    """
    Find the top 10 candidate images from the database.
    Parameters:
        global_descriptor: Global descriptor of the query image.
        db_global_descriptors: Global descriptors of the database images.
        db_image_names: Names of the database images.
    """
    similarity_scores = torch.einsum('id,jd->ij', global_descriptor['global_descriptor'][None, :], db_global_descriptors)
    topk = torch.topk(similarity_scores, 10, dim=1)
    nearest_candidate_images = db_image_names[topk.indices[0].cpu().numpy()]
    nearest_image_descriptors = db_global_descriptors[topk.indices[0]]
    return nearest_candidate_images, nearest_image_descriptors

def get_local_matches(db_local_features_path, 
                      nearest_candidate_images, 
                      local_features, 
                      matcher_model, 
                      query_image_name, 
                      device):
    """
    Find local matches between the query image and the candidate images.
    Parameters:
        db_local_features_path: Path to the database local features.
        nearest_candidate_images: Names of the candidate images.
        local_features: Local features of the query image.
        matcher_model: Local feature matcher model.
        query_image_name: Name of the query image.
        device: Device to run the model on.
    """
    ## Matching image using the image pairs - Optimized
    local_matches = {}
    with h5py.File(db_local_features_path, 'r') as db_local_features:
        for image_name in nearest_candidate_images:
            data = {}

            for k in ['keypoints', 'scores', 'descriptors']:
                data[k + '0'] = local_features[k]
            data['image0'] = torch.empty((1,)+tuple(local_features['image_size'])[::-1], device = device)

            for k in ['keypoints', 'scores', 'descriptors']:
                v = db_local_features[image_name][k]
                data[k + '1'] = torch.from_numpy(v.__array__()).float().to(device)
            data['image1'] = torch.empty((1,)+tuple(db_local_features[image_name]['image_size'])[::-1], device = device)

            for k in data:
                data[k] = data[k].unsqueeze(0)

            with torch.no_grad():
                match = matcher_model(data)
            # breakpoint()
            local_matches[names_to_pair(query_image_name,image_name)] = match

    return local_matches

def _get_matches_from_tensor(local_matches, name0, name1):
    pair_index = names_to_pair(name0, name1)
    matches = local_matches[pair_index]['matches0'].squeeze().detach().cpu().numpy()
    scores = local_matches[pair_index]['matching_scores0'].squeeze().detach().cpu().numpy()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    scores = scores[idx]
    return matches, scores

def get_pose(query_processing_data_dir, 
             query_image_name, 
             db_reconstruction, 
             nearest_candidate_images, 
             local_matches, 
             local_features):
    ## Now we have global candidate and thier mathces. We use this, along with SfM reconstruction to localize the image.
    reconstruction = pycolmap.Reconstruction(db_reconstruction.__str__())
    query_camera = pycolmap.infer_camera_from_image(query_processing_data_dir / query_image_name)
    ref_ids = []
    for r in nearest_candidate_images:
        image = reconstruction.find_image_with_name(r)
        if image is not None:  # Check if the image actually exists in the reconstruction
            ref_ids.append(image.image_id)
            
    conf = {
        'estimation': {'ransac': {'max_error': 12}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(reconstruction, conf)

    # pose from cluster
    kqp = local_features['keypoints'].cpu().numpy()
    kqp += 0.5 # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(ref_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D() == 0:
            print(f'No 3D points found for {image.name}.')
            continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                for p in image.points2D])
        this_match, _ = _get_matches_from_tensor(local_matches, query_image_name, image.name)
        this_match = this_match[points3D_ids[this_match[:, 1]] != -1]
        num_matches += len(this_match)
        for idx, m in this_match:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kqp, mkp_idxs, mp3d_ids, query_camera)
    ret['camera'] = {
        'model': query_camera.model_name,
        'width': query_camera.width,
        'height': query_camera.height,
        'params': query_camera.params,
    }

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                    for i in idxs for j in kp_idx_to_3D[i]]
    log = {
        'db': ref_ids,
        'PnP_ret': ret,
        'keypoints_query': kqp[mkp_idxs],
        'points3D_ids': mp3d_ids,
        'points3D_xyz': None,  # we don't log xyz anymore because of file size
        'num_matches': num_matches,
        'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def copy_file(src, dst):
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())


def dynamic_update(
    query_processing_data_dir,
    query_image_name,
    db_local_features_path,
    local_features,
    db_global_descriptors_path,
    global_descriptor,
    db_reconstruction,
    pose_ret,
    pose_log):


    # Create copies of the local feature and global descriptor .h5 files
    local_features_copy_path = Path(db_local_features_path).parent / f'copy_{Path(db_local_features_path).name}'
    copy_file(db_local_features_path, local_features_copy_path)

    global_descriptors_copy_path = Path(db_global_descriptors_path).parent / f'copy_{Path(db_global_descriptors_path).name}'
    copy_file(db_global_descriptors_path, global_descriptors_copy_path)

    # Update the local feature database
    with h5py.File(local_features_copy_path, 'a') as f:
        grp = f.create_group(query_image_name)
        grp.create_dataset('descriptors', data=local_features['descriptors'])
        grp.create_dataset('image_size', data=local_features['image_size'])
        grp.create_dataset('keypoints', data=local_features['keypoints'])
        grp.create_dataset('scores', data=local_features['scores'])


    # Update the global feature database
    with h5py.File(global_descriptors_copy_path, 'a') as f:
        grp = f.create_group(query_image_name)
        grp.create_dataset('global_descriptor', data=global_descriptor['global_descriptor'])
        grp.create_dataset('image_size', data=global_descriptor['image_size'])


    # Add the image, points3D, and camera (if applicable) to the reconstruction
    cameras, images, points3D = read_model(db_reconstruction)

    query_camera = pycolmap.infer_camera_from_image(query_processing_data_dir / query_image_name)

    # Create and add a new camera
    new_camera_id = max(cameras.keys()) + 1
    new_camera = Camera(
        id=new_camera_id,
        model=query_camera.model_name,
        width=query_camera.width,
        height=query_camera.height,
        params=query_camera.params
    )
    cameras[new_camera_id] = new_camera

    # Create and add a new image
    new_image_id = max(images.keys()) + 1
    new_image = Image(
        id=new_image_id,
        qvec=pose_ret['qvec'],
        tvec=pose_ret['tvec'],
        camera_id=new_camera_id,
        name=query_image_name,
        point3D_ids=pose_log['points3D_ids'],
        xys=local_features['keypoints']
    )
    images[new_image_id] = new_image

    model_path = Path(db_reconstruction)
    output_model_path = model_path.parent / 'updated_reconstruction'
    os.makedirs(output_model_path, exist_ok=True)
    write_model(cameras, images, points3D, output_model_path)


def localize(query_processing_data_dir, query_image_name, 
             device, 
             local_feature_conf, local_features_extractor_model, 
             global_descriptor_conf, global_descriptor_model, 
             db_global_descriptors, db_global_descriptors_path, 
             db_image_names, db_local_features_path, matcher_model, 
             db_reconstruction):
    print(f"Called Localize for image{query_processing_data_dir}")
    print("Running get_local_features")
    local_features, uncertainty = get_local_features(
        query_processing_data_dir = query_processing_data_dir, 
        query_image_name = query_image_name,
        local_feature_conf = local_feature_conf,
        local_features_extractor_model = local_features_extractor_model,
        device = device
    )
    print("Finished get_local_features")

    print("Running get_global_descriptors")
    global_descriptor = get_global_descriptors(
        query_processing_data_dir = query_processing_data_dir, 
        query_image_name = query_image_name, 
        global_descriptor_conf = global_descriptor_conf, 
        global_descriptor_model = global_descriptor_model, 
        device = device
    )
    print("Finished get_global_descriptors")

    print("Running get_candidate_matches")
    nearest_candidate_images, nearest_image_descriptors = get_candidate_matches(
        global_descriptor = global_descriptor, 
        db_global_descriptors = db_global_descriptors, 
        db_image_names = db_image_names
    )
    print("Finished get_candidate_matches")

    print("Running get_local_matches")
    local_matches = get_local_matches(
        db_local_features_path = db_local_features_path, 
        nearest_candidate_images = nearest_candidate_images, 
        local_features = local_features, 
        matcher_model = matcher_model, 
        query_image_name = query_image_name, 
        device = device
    )
    print("Finished get_local_matches")

    print("Running get_pose")
    ret, log = get_pose(
        query_processing_data_dir = query_processing_data_dir, 
        query_image_name = query_image_name, 
        db_reconstruction = db_reconstruction, 
        nearest_candidate_images = nearest_candidate_images, 
        local_matches = local_matches, 
        local_features = local_features
    )
    print("Finished get_pose: ", ret['qvec'], ret['tvec'])

    # dynamic_update(
    #     query_processing_data_dir = query_processing_data_dir,
    #     query_image_name = query_image_name,
    #     db_local_features_path = db_local_features_path,
    #     local_features = local_features,
    #     db_global_descriptors_path = db_global_descriptors_path,
    #     global_descriptor = global_descriptor,
    #     db_reconstruction = db_reconstruction,
    #     pose_ret = ret,
    #     pose_log = log)

    def homogenize(rotation, translation):
        """
        Combine the (3,3) rotation matrix and (3,) translation matrix to
        one (4,4) transformation matrix
        """
        homogenous_array = np.eye(4)
        homogenous_array[:3, :3] = rotation
        homogenous_array[:3, 3] = translation
        return homogenous_array

    def rot_from_qvec(qvec):
        # Change (w,x,y,z) to (x,y,z,w)
        return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])

    reconstruction = pycolmap.Reconstruction(db_reconstruction.__str__())
    ref_imgs = [reconstruction.find_image_with_name(r) for r in nearest_candidate_images]
    for i, db_img in enumerate(ref_imgs):
        this_match, _ = _get_matches_from_tensor(local_matches, query_image_name, db_img.name)
        query_pose = np.linalg.inv(homogenize(rot_from_qvec(ret['qvec']).as_matrix(), ret['tvec']))[:-1]
        ref_pose = np.linalg.inv(homogenize(rot_from_qvec(db_img.qvec).as_matrix(), db_img.tvec))[:-1]
        print(db_img.name)
        query_points = []
        ref_points = []
        for query_kp, db_kp in this_match:
            q = local_features['keypoints'][query_kp]
            query = np.array([q[0].item(),q[1].item()])
            point2D = db_img.points2D[db_kp]
            ref = np.array(point2D.xy)
            if point2D.point3D_id == 18446744073709551615:
                query_points.append(query)
                ref_points.append(ref)
        print("query_pose:")
        print(query_pose)
        print("ref_pose:")
        print(ref_pose)
        print("query_pose:")
        print(np.array(query_points).T)
        print("ref_points:")
        print(np.array(ref_points).T)
        if len(query_points) > 0:
            point4D = cv2.triangulatePoints(query_pose, ref_pose, np.array(query_points).T, np.array(ref_points).T)
            point3D = point4D[:3] / point4D[3]
            point3D = np.array(point3D).T
            print("point3D:")
            print(point3D)

    return ret, log
