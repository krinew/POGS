import warp as wp
import torch
from nerfstudio.cameras.cameras import Cameras
from pogs.tracking.transforms import SE3, SO3

def extrapolate_poses(p1_7v, p2_7v, lam, rot_scale_factor=3.0, thresh=True):
    ext_7v = []
    for i in range(len(p2_7v)):
        r1 = SO3(p1_7v[i,3:])
        t1 = SE3.from_rotation_and_translation(r1, p1_7v[i,:3])
        r2 = SO3(p2_7v[i,3:])
        t2 = SE3.from_rotation_and_translation(r2, p2_7v[i,:3])
        t_2_1 = t1.inverse() @ t2
        delta_pos = t_2_1.translation()*lam
        delta_rot = SO3.exp((t_2_1.rotation().log() * lam * rot_scale_factor))
        if thresh and delta_pos.norm().item() < 0.05: # Threshold for small deltas to avoid oscillations
            new_t = t2
        else:
            new_t = (t2 @ SE3.from_rotation_and_translation(delta_rot, delta_pos))
        ext_7v.append(new_t.wxyz_xyz.roll(3,dims=-1))
    return torch.stack(ext_7v)

def zero_optim_state(optimizer:torch.optim.Adam):
    # import pdb; pdb.set_trace()
    param = optimizer.param_groups[0]["params"][0]
    param_state = optimizer.state[param]
    if "max_exp_avg_sq" in param_state:
        # for amsgrad
        param_state["max_exp_avg_sq"] = torch.zeros(param_state["max_exp_avg_sq"].shape, device=param_state["max_exp_avg_sq"].device)
    if "exp_avg" in param_state:
        param_state["exp_avg"] = torch.zeros(param_state["exp_avg"].shape, device=param_state["exp_avg"].device)
        param_state["exp_avg_sq"] = torch.zeros(param_state["exp_avg_sq"].shape, device=param_state["exp_avg_sq"].device)

def replace_in_optim(optimizer:torch.optim.Adam, new_params):
    """replaces the parameters in the optimizer"""
    param = optimizer.param_groups[0]["params"][0]
    param_state = optimizer.state[param]

    del optimizer.state[param]
    optimizer.state[new_params[0]] = param_state
    optimizer.param_groups[0]["params"] = new_params
    del param

@wp.func
def poses_7vec_to_transform(poses: wp.array(dtype=float, ndim=2), i: int):
    """
    Kernel helper for converting x y z qw qx qy qz to a wp.Transformation
    """
    position = wp.vector(poses[i,0], poses[i,1], poses[i,2])
    quaternion = wp.quaternion(poses[i,4], poses[i,5], poses[i,6], poses[i,3])
    return wp.transformation(position, quaternion)

@wp.kernel
def apply_to_model(
    # init_o2w: wp.array(dtype=float, ndim=2),
    init_p2ws: wp.array(dtype=float, ndim=2),
    # o_delta: wp.array(dtype=float, ndim=2),
    p_deltas: wp.array(dtype=float, ndim=2),
    group_labels: wp.array(dtype=int),
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=float, ndim=2),
    #outputs
    means_out: wp.array(dtype=wp.vec3),
    quats_out: wp.array(dtype=float, ndim=2),
):
    """
    Kernel for applying the transforms to a gaussian splat

    [removed] init_o2w: 1x7 tensor of initial object to world poses
    init_p2ws: Nx7 tensor of initial pose to object poses
    [removed] o_delta: Nx7 tensor of object pose deltas represented as objnew_to_objoriginal
    p_deltas: Nx7 tensor of pose deltas represented as partnew_to_partoriginal
    group_labels: N, tensor of group labels (0->K-1) for K groups
    means: Nx3 tensor of means
    quats: Nx4 tensor of quaternions (wxyz)
    means_out: Nx3 tensor of output means
    quats_out: Nx4 tensor of output quaternions (wxyz)
    """
    tid = wp.tid()
    group_id = group_labels[tid]
    # o2w_T = poses_7vec_to_transform(init_o2w,0)
    p2w_T = poses_7vec_to_transform(init_p2ws,group_id)
    # odelta_T = poses_7vec_to_transform(o_delta,0)
    pdelta_T = poses_7vec_to_transform(p_deltas,group_id)
    g2w_T = wp.transformation(means[tid], wp.quaternion(quats[tid, 1], quats[tid, 2], quats[tid, 3], quats[tid, 0]))
    g2p_T = wp.transform_inverse(p2w_T) * g2w_T
    new_g2w_T = p2w_T * pdelta_T * g2p_T
    means_out[tid] = wp.transform_get_translation(new_g2w_T)
    new_quat = wp.transform_get_rotation(new_g2w_T)
    quats_out[tid, 0] = new_quat[3] #w
    quats_out[tid, 1] = new_quat[0] #x
    quats_out[tid, 2] = new_quat[1] #y
    quats_out[tid, 3] = new_quat[2] #z

def identity_7vec(device='cuda'):
    """
    Returns a 7-tensor of identity pose
    """
    return torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=torch.float32, device=device)

def normalized_quat_to_rotmat(quat):
    """
    Converts a quaternion to a 3x3 rotation matrix
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))

def torch_posevec_to_mat(posevecs):
    """
    Converts a Nx7-tensor to Nx4x4 matrix

    posevecs: Nx7 tensor of pose vectors
    returns: Nx4x4 tensor of transformation matrices
    """
    assert posevecs.shape[-1] == 7, posevecs.shape
    assert len(posevecs.shape) == 2, posevecs.shape
    out = torch.eye(4, device=posevecs.device).unsqueeze(0).expand(posevecs.shape[0], -1, -1)
    out[:, :3, 3] = posevecs[:, :3]
    out[:, :3, :3] = normalized_quat_to_rotmat(posevecs[:, 3:])
    return out

def mnn_matcher(feat_a, feat_b):
    """
    Returns mutual nearest neighbors between two sets of features

    feat_a: NxD
    feat_b: MxD
    return: K, K (indices in feat_a and feat_b)
    """
    device = feat_a.device
    sim = feat_a.mm(feat_b.t())
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    return ids1[mask], nn12[mask]

def crop_camera(camera: Cameras, xmin, xmax, ymin, ymax):
    height = torch.tensor(ymax - ymin,device='cuda').view(1,1).int()
    width = torch.tensor(xmax - xmin,device='cuda').view(1,1).int()
    cx = torch.tensor(camera.cx.clone() - xmin,device='cuda').view(1,1)
    cy = torch.tensor(camera.cy.clone() - ymin,device='cuda').view(1,1)
    fx = camera.fx.clone()
    fy = camera.fy.clone()
    return Cameras(camera.camera_to_worlds.clone(), fx, fy, cx, cy, width, height)